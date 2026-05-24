import os
import sys
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import config
from src.model import resnet18, SimpleCNN
from src.utils import get_device, load_model

CLASS_NAMES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# Model and transforms
@st.cache_resource
def load_trained_model():
    # Try ResNet first, then fallback to SimpleCNN if the checkpoint was trained with a different architecture.
    model = resnet18(num_classes=config.NUM_CLASSES)
    try:
        if load_model(model, "best_model.pt"):
            model.eval()
            return model
    except RuntimeError:
        pass

    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    try:
        if load_model(model, "best_model.pt"):
            model.eval()
            return model
    except RuntimeError:
        pass

    return None

def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

def preprocess(image: Image.Image):
    image = image.convert('RGB')
    tensor = get_transform()(image).unsqueeze(0)
    return tensor

def predict(image: Image.Image):
    model = load_trained_model()
    if model is None:
        return None, None, None

    device = get_device()
    model.to(device)

    input_tensor = preprocess(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()

    top_indices = probabilities.argsort()[::-1][:3]
    top_probs = probabilities[top_indices]
    top_labels = [CLASS_NAMES[idx] for idx in top_indices]
    return top_labels, top_probs, probabilities

# Streamlit UI
st.set_page_config(
    page_title='CIFAR-10 Classifier',
    page_icon='📷',
    layout='centered',
)

st.title('CIFAR-10 Image Classification')
st.write(
    'Upload an image and use the trained ResNet-18 model to predict the CIFAR-10 class.'
)

st.sidebar.header('Instructions')
st.sidebar.write(
    '1. Upload a photo (RGB or PNG/JPG).\n'
    '2. Click the button to run inference.\n'
    '3. The app will show the top predictions and their confidence scores.'
)

model = load_trained_model()
if model is None:
    st.error(
        'Không tìm thấy model đã huấn luyện `models/best_model.pt`.\n'
        'Vui lòng huấn luyện model trước bằng `python main.py train` hoặc đặt file model vào thư mục `models/`.'
    )
    st.stop()

uploaded_file = st.file_uploader('Chọn ảnh để dự đoán', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đầu vào', use_column_width=True)

    if st.button('Dự đoán'):  
        with st.spinner('Đang suy luận...'):
            labels, probs, all_probs = predict(image)

        if labels is None:
            st.error('Không thể load model. Kiểm tra lại file `models/best_model.pt`.')
        else:
            st.subheader('Kết quả dự đoán')
            for label, prob in zip(labels, probs):
                st.write(f'- **{label}**: {prob * 100:.2f}%')

            st.markdown('---')
            st.subheader('Top 10 xác suất')
            result_table = {
                'Class': [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
                'Confidence': [f'{p * 100:.2f}%' for p in all_probs]
            }
            st.table(result_table)

st.markdown('---')
st.write('Phiên bản app: `streamlit_app.py`. Chạy bằng: `streamlit run streamlit_app.py`.')
