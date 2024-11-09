from math import pi
from taipy.gui import Gui, notify, download
from PIL import Image
from io import BytesIO
import taipy.gui.builder as tgb
import pickle
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
<<<<<<< HEAD
import gdown

=======
from tensorflow.keras.utils import load_img, img_to_array
>>>>>>> parent of 412c605 (upload models)

path_upload = ""
original_image = None
image = None
colorized_image = None
colorized = False

style_image_path = ""
content_image_path = ""
styled_image = None
styled = False
style_image = None
content_image = None


# Load the colorization model
model_url = 'https://drive.google.com/uc?id=1_UuFP93RXET8evufhCt7sTuDXUPpj8tk'
gdown.download(model_url, 'colorizer_model.pkl', quiet=False)

with open("colorizer_model.pkl", "rb") as model_file:
    colorizer_model = pickle.load(model_file)

# Load the styling model     
<<<<<<< HEAD
style_model = hub.load('https://kaggle.com/models/google/arbitrary-image-stylization-v1/frameworks/TensorFlow1/variations/256/versions/1')
# with open("styl_model.pkl", "rb") as model_file:
#     style_model = pickle.load(model_file)

=======
style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# with open("style_model.pkl", "rb") as style_model_file:
#     style_model = pickle.load(style_model_file)
>>>>>>> parent of 412c605 (upload models)

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def preprocess_image(img):
    original_size = img.size
    img_array = np.array(img.resize((256, 256)))
    img_array = img_array / 255.0
    img_array = np.repeat(img_array[:, :, np.newaxis], 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_size

def upload_image(state):
    state.image = Image.open(state.path_upload).convert('L')
    state.original_image = convert_image(state.image)
    state.colorized = False

def colorize_image(state, id=None, action=None):
    state.colorized = False
    notify(state, "info", "Colorizing the image...")
    
    preprocessed_image, original_size = preprocess_image(state.image)
    predicted_image = colorizer_model.predict(preprocessed_image)
    output_image = Image.fromarray((predicted_image[0] * 255).astype('uint8'))
    output_image = output_image.resize(original_size, Image.LANCZOS)
    
    state.colorized_image = convert_image(output_image)
    state.colorized = True
    notify(state, "success", "Image colorized successfully!")

def upload_style_image(state):
    state.style_image = Image.open(state.style_image_path)
    state.styled = False

def upload_content_image(state):
    state.content_image = Image.open(state.content_image_path)
    state.styled = False

def style_image(state, id=None, action=None):
    if state.style_image is None or state.content_image is None:
        notify(state, "error", "Please upload both style and content images.")
        return

    state.styled = False
    notify(state, "info", "Styling the image...")
    
    style_preprocessed, _ = preprocess_image(state.style_image)
    content_preprocessed, original_size = preprocess_image(state.content_image)
    
    styled_output = style_model(tf.constant(content_preprocessed), tf.constant(style_preprocessed))
    styled_image = Image.fromarray(tf.cast(styled_output[0] * 255, tf.uint8).numpy())
    styled_image = styled_image.resize(original_size, Image.LANCZOS)
    
    state.styled_image = convert_image(styled_image)
    state.styled = True
    notify(state, "success", "Image styled successfully!")

def download_image(state):
    if state.colorized:
        download(state, content=state.colorized_image, name="colorized_img.png")
    elif state.styled:
        download(state, content=state.styled_image, name="styled_img.png")

def colorization_tab():
    with tgb.part():
        tgb.text("# Image **Colorizer**", mode="md")
        tgb.text(
            """
Upload a grayscale image to see it transformed into a vibrant, colorized version. Download the full-quality colorized image from the sidebar.
""",
            mode="md",
        )
        with tgb.layout("1 1"):
            with tgb.part("card text-center", render="{original_image}"):
                tgb.text("### Original Grayscale Image 📷", mode="md")
                tgb.image("{original_image}")
            with tgb.part("card text-center", render="{colorized_image}"):
                tgb.text("### Colorized Image 🎨", mode="md")
                tgb.image("{colorized_image}")

def styling_tab():
    with tgb.part():
        tgb.text("# Image **Styler**", mode="md")
        tgb.text("Upload a style image and a content image to combine them.", mode="md")
        with tgb.layout("1 1 1"):
            with tgb.part("card text-center", render="{style_image_path}"):
                tgb.text("### Style Image 🖼️", mode="md")
                tgb.image("{style_image_path}", style="max-width: 100%; height: auto;")
            with tgb.part("card text-center", render="{content_image_path}"):
                tgb.text("### Content Image 📷", mode="md")
                tgb.image("{content_image_path}", style="max-width: 100%; height: auto;")
            with tgb.part("card text-center", render="{styled_image}"):
                tgb.text("### Styled Image 🎨", mode="md")
                tgb.image("{styled_image}", style="max-width: 100%; height: auto;")

with tgb.Page() as page:
    tgb.toggle(theme=True)

    with tgb.layout("20 80", columns__mobile="1"):
        with tgb.part("sidebar"):
            tgb.text("### Image Processing", mode="md")
            tgb.file_selector(
                "{path_upload}",
                extensions=".png,.jpg",
                label="Upload image for colorization",
                on_action=upload_image,
                class_name="fullwidth",
            )
            tgb.button("Colorize", on_action=colorize_image)
            
            tgb.file_selector(
                "{style_image_path}",
                extensions=".png,.jpg",
                label="Upload style image",
                on_action=upload_style_image,
                class_name="fullwidth",
            )
            tgb.file_selector(
                "{content_image_path}",
                extensions=".png,.jpg",
                label="Upload content image",
                on_action=upload_content_image,
                class_name="fullwidth",
            )
            tgb.button("Style", on_action=style_image)
            
            tgb.file_download(
                "{colorized_image}|{styled_image}",
                label="Download processed image",
                on_action=download_image,
                active="{colorized}|{styled}",
                class_name="fullwidth",
            )

        with tgb.part("container"):
            with tgb.layout("1"):
                with tgb.part():
                    colorization_tab()
                with tgb.part():
                    styling_tab()

if __name__ == "__main__":
    Gui(page=page).run(margin="0px", title="Image Processor")
