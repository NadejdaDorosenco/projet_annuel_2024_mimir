from flask import Flask, request, render_template, session, redirect, url_for, send_file, flash
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # type:ignore
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type:ignore
import numpy as np
import cv2
import os
import time
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from google.cloud import storage
import tempfile

from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

# Initialise le client Google Cloud Storage
#credentials = service_account.Credentials.from_service_account_file('bucket_key.json')
#storage_client = storage.Client(credentials=credentials)
storage_client = storage.Client()
bucket = storage_client.get_bucket('data-train-mimir')

# Configuration pour le téléchargement de fichiers
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Fonction pour vérifier l'extension de fichier
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_from_gcs(model_file_path):
    blob = bucket.blob(model_file_path)

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        model = tf.keras.models.load_model(temp_file.name)
        print('model: ', model)
    
    # Tenter de supprimer le fichier temporaire après la fermeture du fichier
    try:
        os.remove(temp_file.name)
    except PermissionError as e:
        print(f"Erreur lors de la suppression du fichier temporaire: {e}")

    return model

# Charger les modèles
global_model = load_model_from_gcs('models/ResNet50_global.h5')
binary_model_1 = load_model_from_gcs('models/ResNet50_binaire_1.h5')
binary_model_2 = load_model_from_gcs('models/ResNet50_binaire_2.h5')
binary_model_3 = load_model_from_gcs('models/ResNet50_binaire_3.h5')

# Classes
global_classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
binary_classes = {
    'NonDemented_VeryMildDemented': ['NonDemented', 'VeryMildDemented'],
    'VeryMildDemented_MildDemented': ['MildDemented', 'VeryMildDemented'],
    'MildDemented_ModerateDemented': ['MildDemented', 'ModerateDemented']
}

# Dictionnaire de traduction
translations = {
    'NonDemented': 'Pas de démence',
    'VeryMildDemented': 'Démence très légère',
    'MildDemented': 'Démence légère',
    'ModerateDemented': 'Démence modérée'
}

# Fonction pour traduire les classes
def translate_class(class_name):
    return translations.get(class_name, class_name)

def prepare_image(image_path):
    img = load_img(image_path, target_size=(176, 176))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    resnet50_model = model.get_layer('resnet50')
    last_conv_layer = resnet50_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(resnet50_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    if tf.reduce_max(heatmap) == 0:
        print("Attention: la carte de chaleur est nulle pour cette image.")
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionne la carte de chaleur pour qu'elle corresponde à la taille de l'image d'origine
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convertir la carte de chaleur en une échelle de 0 à 255
    heatmap = np.uint8(255 * heatmap)
    
    # Applique la colormap 'JET' pour obtenir les couleurs chaudes
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superpose la carte de chaleur sur l'image d'origine
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    # Sauvegarde l'image superposée
    heatmap_path = os.path.join('static', 'heatmap_image.jpg')
    cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    return superimposed_img, heatmap_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Aucun fichier sélectionné")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("Aucun fichier sélectionné")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if not os.path.exists('static'):
                os.makedirs('static')

            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)
            img_array = prepare_image(file_path)

            # Prédiction avec le modèle global
            start_time = time.time()
            global_pred = global_model.predict(img_array)
            end_time = time.time()
            prediction_time = end_time - start_time

            global_class = global_classes[np.argmax(global_pred)]
            top3_classes = np.argsort(global_pred[0])[-3:][::-1]
            top3_scores = global_pred[0][top3_classes]
            top3_results = [(global_classes[i], round(float(top3_scores[idx]) * 100, 2)) for idx, i in enumerate(top3_classes)]

            # Debugging: Affiche les prédictions brutes
            print("Global prediction scores:", global_pred[0])
            print("Top 3 classes and scores:", top3_results)

            session['file_path'] = file_path
            session['global_class'] = global_class
            session['prediction_time'] = round(float(prediction_time), 2)
            session['top3_results'] = top3_results

            # Génération de la carte de chaleur Grad-CAM
            last_conv_layer_name = 'conv5_block3_out'
            classifier_layer_names = [layer.name for layer in global_model.layers if 'resnet' not in layer.name]
            heatmap = make_gradcam_heatmap(img_array, global_model, last_conv_layer_name, classifier_layer_names)
            _, heatmap_path = save_and_display_gradcam(file_path, heatmap)
            session['heatmap_path'] = heatmap_path

            # Traduit la classe globale avant de l'envoyer au modèle
            translated_global_class = translate_class(global_class)

            # Affiche result_global.html avec la prédiction globale
            return render_template('result_global.html', 
                                   global_class=translated_global_class, 
                                   prediction_time=session['prediction_time'],
                                   top3_results=top3_results,
                                   heatmap_path=heatmap_path,
                                   file_path=file_path,
                                   translations=translations)
        else:
            flash("Type de fichier non autorisé. Veuillez télécharger une image (png, jpg, jpeg, gif).")
            return redirect(request.url)

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file_path = session.get('file_path', None)
    global_class = session.get('global_class', None)
    prediction_time = session.get('prediction_time', None)
    top3_results = session.get('top3_results', [])
    heatmap_path = session.get('heatmap_path', None)

    if not file_path or not global_class:
        return "Session expirée. Veuillez télécharger l'image à nouveau."

    img_array = prepare_image(file_path)
    selected_model = request.form.get('model')
    if selected_model == 'binary_model_1':
        binary_model = binary_model_1
        binary_class_names = binary_classes['NonDemented_VeryMildDemented']
    elif selected_model == 'binary_model_2':
        binary_model = binary_model_2
        binary_class_names = binary_classes['VeryMildDemented_MildDemented']
    elif selected_model == 'binary_model_3':
        binary_model = binary_model_3
        binary_class_names = binary_classes['MildDemented_ModerateDemented']
    else:
        return "Sélection de modèle invalide."

    binary_pred = binary_model.predict(img_array)
    print("Binary prediction scores:", binary_pred[0])  # Debugging: Affiche la prédiction binaire brute

    # Assure que la prédiction est basée sur la plus grande probabilité
    max_index = np.argmax(binary_pred[0])
    binary_class = binary_class_names[max_index]
    binary_pred_value = round(float(binary_pred[0][max_index]) * 100, 2)

    return render_template('result_binaire.html', 
                           global_class=translate_class(global_class), 
                           binary_class=translate_class(binary_class),
                           binary_pred_value=binary_pred_value, 
                           prediction_time=prediction_time,
                           top3_results=[(translate_class(cls), score) for cls, score in top3_results], 
                           heatmap_path=heatmap_path,
                           file_path=file_path)

@app.route('/save_image', methods=['POST'])
def save_image():
    file_path = session.get('file_path', None)
    save_location = request.form.get('save_location', None)
    image_name = request.form.get('image_name', None)
    
    if not file_path or not save_location or not image_name:
        return "Requête de sauvegarde invalide. Veuillez revenir en arrière et réessayer."
    
    bucket_path = f"patient-images/{save_location}/{image_name}"
    
    # On s'assure que le nom de l'image a l'extension de fichier correcte
    _, file_extension = os.path.splitext(file_path)
    if not image_name.endswith(file_extension):
        image_name += file_extension
    
    # Lis l'image depuis le système de fichiers local
    with open(file_path, "rb") as image_file:
        blob = bucket.blob(bucket_path)
        blob.upload_from_file(image_file)
    
    return redirect(url_for('index'))

@app.route('/download_report')
def download_report():
    file_path = session.get('file_path', None)
    global_class = session.get('global_class', None)
    prediction_time = session.get('prediction_time', None)
    top3_results = session.get('top3_results', [])
    heatmap_path = session.get('heatmap_path', None)

    if not file_path or not global_class or not prediction_time or not heatmap_path:
        return "Données de session manquantes. Veuillez réessayer."

    # Crée le répertoire de sauvegarde s'il n'existe pas
    if not os.path.exists(os.path.join('static', 'reports')):
        os.makedirs(os.path.join('static', 'reports'))

    # Crée un fichier PDF
    report_path = os.path.join('static', 'reports', 'report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Titre du rapport
    elements.append(Paragraph("Rapport de Prédiction", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    # Traduit la classe globale prédite
    translated_global_class = translate_class(global_class)
    elements.append(Paragraph(f"Classe globale prédite: {translated_global_class}", styles['Heading2']))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(f"Temps de prédiction: {prediction_time} secondes", styles['BodyText']))
    elements.append(Spacer(1, 0.2 * inch))

    # Top 3 des prédictions globales
    elements.append(Paragraph("Top 3 des prédictions globales:", styles['Heading2']))
    elements.append(Spacer(1, 0.1 * inch))
    data = [["Classe", "Confiance"]] + [[translate_class(cls), f"{score}%"] for cls, score in top3_results]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2 * inch))

    # Carte de chaleur
    elements.append(Paragraph("Carte de chaleur:", styles['Heading2']))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(heatmap_path, width=4*inch, height=4*inch))

    doc.build(elements)

    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))