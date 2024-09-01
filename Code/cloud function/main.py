from cloudevents.http import CloudEvent
import functions_framework
import os
import shutil
from google.cloud import storage

bucket_name = 'data-train-mimir'
IMAGE_THRESHOLD = 2  # défini le nombre d'images sauvegardés dans patient_images à partir duquel on réentraine les models

# Fonction pour nettoyer un répertoire
def clean_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f'Nettoyé le répertoire : {directory}')
    os.makedirs(directory)
    print(f'Créé le répertoire : {directory}')

# Fonction pour télécharger les blobs depuis GCS et préserver la structure des répertoires
def download_blobs(bucket_name, source_blob_folder, destination_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_folder)

    for blob in blobs:
        if not blob.name.endswith('/'):
            # Créer le chemin local correspondant au blob
            destination_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_blob_folder))
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            blob.download_to_filename(destination_path)
            print(f'Downloaded {blob.name} to {destination_path}')

# Fonction pour vérifier et copier les images si le seuil est atteint
def check_and_copy_images(bucket_name, source_folder, destination_folder, threshold):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=source_folder))

    if len(blobs) >= threshold:
        for blob in blobs:
            if not blob.name.endswith('/'):
                new_blob_name = blob.name.replace(source_folder, destination_folder) + '_patient_image'
                bucket.copy_blob(blob, bucket, new_blob_name)
                print(f'Copied {blob.name} to {new_blob_name}')
                blob.delete()
                print(f'Deleted {blob.name} from {source_folder}')
                
@functions_framework.cloud_event
def collect_data(cloud_event: CloudEvent):
    """This function is triggered by a change in a storage bucket.

    Args:
        cloud_event: The CloudEvent that triggered this function.
    """
    data = cloud_event.data
    bucket_name = data['bucket']
    source_blob_name = data['name']

    # Filtre les fichiers par répertoire
    if source_blob_name.startswith('patient-images/'):
        # Vérifie et copie les images si le seuil est atteint
        check_and_copy_images(bucket_name, 'patient-images/', 'data/train/', IMAGE_THRESHOLD)
        print('Checked and copied images if threshold exceeded.')
    else:
        print(f"Ignoring file {source_blob_name} as it's not in the expected directories.")

if __name__ == '__main__':
    # Test local (ne sera pas utilisé lors du déploiement)
    event = CloudEvent({
        "type": "google.storage.object.finalize",
        "source": "//storage.googleapis.com/projects/_/buckets/data-train-mimir",
    }, {
        "bucket": "data-train-mimir",
        "name": "patient-images/Test.jpg",
        "contentType": "image/jpeg",
        "metageneration": "1",
        "timeCreated": "2023-04-23T07:38:57.230Z",
        "updated": "2023-04-23T07:38:57.230Z"
    })

    collect_data(event)
