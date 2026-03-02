import argparse
import datetime
import os

import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.src.optimizers import RMSprop

from utils.callbacks import ReconstructionPlot, CoefficientScheduler, CollapseCallback
from utils.helper import Helper

from model.encoder import Encoder
from model.decoder import Decoder
from model.tcvae import TCVAE

from matplotlib import pyplot as plt
import pandas as pd
import sys

# Set path to the location of the tensorflow datasets
os.environ['TFDS_DATA_DIR'] = '/users/wadh6616/tensorflow_datasets/'

# Limit the number of threads to prevent memory issues on CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Define the target directory
custom_path = r"/users/wadh6616/VAE_ECG/src"
# Add to sys.path
if custom_path not in sys.path:
    sys.path.append(custom_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(parameters):
    ######################################################
    # INITIALIZATION
    ######################################################
    tf.random.set_seed(parameters['seed'])
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_path = parameters['save_results_path'] + start_time + '/'
    Helper.generate_paths(
        [base_path, base_path + 'model_best/', base_path + 'model_final/',
         base_path + 'training/reconstruction/', base_path + 'training/collapse/']
    )
    Helper.write_json_file(parameters, base_path + 'params.json')
    Helper.print_available_gpu()

    ######################################################
    # DATA LOADING
    ######################################################
    train, size_train = Helper.load_multiple_datasets(parameters['train_dataset'])
    val, size_val = Helper.load_multiple_datasets(parameters['val_dataset'])

    ######################################################
    # MACHINE LEARNING
    ######################################################
    callbacks = [
        TerminateOnNaN(),
        CollapseCallback(val, base_path + 'training/collapse/'),
        CSVLogger(base_path + 'training/training_progress.csv'),
        EarlyStopping(monitor="val_loss", patience=parameters['early_stopping']),
        CoefficientScheduler(parameters['epochs'], parameters['coefficients'], parameters['coefficients_raise']),
        ReduceLROnPlateau(monitor='recon', factor=0.05, patience=20, min_lr=0.000001),
        ModelCheckpoint(filepath=base_path + 'model_best/', monitor='loss', save_best_only=True, verbose=0),
        ReconstructionPlot(train[0], base_path + 'training/reconstruction/', parameters['reconstruction']),
    ]

    encoder = Encoder(parameters['latent_dimension'])
    decoder = Decoder(parameters['latent_dimension'])
    vae = TCVAE(encoder, decoder, parameters['coefficients'], size_train)
    vae.compile(optimizer=RMSprop(learning_rate=parameters['learning_rate']))
    vae.fit(
        Helper.data_generator(train), steps_per_epoch=size_train,
        validation_data=Helper.data_generator(val), validation_steps=size_val,
        epochs=parameters['epochs'], callbacks=callbacks, verbose=1,
    )

    vae.save(base_path + 'model_final/')


    ######################################################
    # THOMAS' VISUALISATIONS
    ######################################################
        # Load training log
    log_path = base_path + 'training/training_progress.csv'
    df = pd.read_csv(log_path)

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plot_path = base_path + 'training/loss_curve.png'
    plt.savefig(plot_path)

    csv_path = base_path
    save_path = os.path.join(csv_path,'kl_loss')
    csv_path =os.path.join(csv_path,'training_progress.csv')
    plt.close()
    plt.figure(figsize=(12, 8))
    plt.plot(df['epoch'], df['kl_loss'], label='Total KL Loss')
    plt.plot(df['epoch'], df['mi'], label='Mutual Information (MI)')
    plt.plot(df['epoch'], df['tc'], label='Total Correlation (TC)')
    plt.plot(df['epoch'], df['dw_kl'], label='Dimension-wise KL')
    plt.plot(df['epoch'], df['recon'], label='Reconstruction Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Component Value')
    plt.title('KL Divergence Components and Reconstruction Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

    print(f"📉 Loss curve saved at: {plot_path}")

    # 🚀 2️⃣ Try Saving the Model Properly
    model_path = base_path + "model_final.keras"

    
    try:
        vae.save(model_path)
        print(f"\n✅ Model saved successfully at: {model_path}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")

    ######################################################
    # END OF THOMAS' VISUALISATIONS
    ######################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='VECG', description='Representational Learning of ECG using disentangling VAE',
    )
    parser.add_argument(
        '-p', '--path_config', type=str, default='./params.yml',
        help='location of the params file (default: ./params.yml)',
    )

    args = parser.parse_args()
    parameters = Helper.load_yaml_file(args.path_config)

    combinations = [
        # Your working baselines
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 1.3, 'gamma': 0.7}},  # worked well
        {'latent_dimension': 64, 'coefficients': {'alpha': 1.5, 'beta': 0.7, 'gamma': 0.1}}, 

        # Push beta higher — more disentanglement pressure
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 2.0, 'gamma': 0.7}},
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 3.0, 'gamma': 0.7}},
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 4.0, 'gamma': 0.7}},

        # Lower alpha — less MI penalty, more information into z
        {'latent_dimension': 64, 'coefficients': {'alpha': 1.0, 'beta': 1.3, 'gamma': 0.7}},
        {'latent_dimension': 64, 'coefficients': {'alpha': 0.5, 'beta': 1.3, 'gamma': 0.7}},

        # Higher gamma — more pressure on each dim to match prior
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 1.3, 'gamma': 1.0}},
        {'latent_dimension': 64, 'coefficients': {'alpha': 2.0, 'beta': 1.3, 'gamma': 2.0}},

        # All low — close to standard VAE, minimal disentanglement pressure
        {'latent_dimension': 64, 'coefficients': {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.1}},

        # Balanced — equal weight on all three terms
        {'latent_dimension': 64, 'coefficients': {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}}
        ]

    for k in combinations:
        parameters.update(k)
        main(parameters)
