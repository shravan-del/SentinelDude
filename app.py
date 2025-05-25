import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Using YOLOv8 nano model

def calculate_threat_score(results, image_size):
    """Calculate threat score based on object proximity and size"""
    threat_score = 0
    detected_objects = []
    
    # Process detection results
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Calculate metrics
            area = (x2 - x1) * (y2 - y1) / (image_size[0] * image_size[1])
            center_x = (x1 + x2) / 2 / image_size[0]
            center_y = (y1 + y2) / 2 / image_size[1]
            
            # Calculate distance from center
            distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            
            # Object specific threat calculation
            object_threat = conf * (1 - distance_from_center) * (area * 100)
            threat_score += object_threat
            
            detected_objects.append({
                'class': results[0].names[cls],
                'confidence': round(conf * 100, 2),
                'area_percent': round(area * 100, 2),
                'threat_contribution': round(object_threat, 2)
            })
    
    return min(100, threat_score * 100), detected_objects

def plot_threat_analysis(detected_objects):
    if not detected_objects:
        return None
    
    df = pd.DataFrame(detected_objects)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['class'], df['threat_contribution'])
    
    # Customize the plot
    ax.set_title('Threat Analysis by Object Type')
    ax.set_xlabel('Detected Objects')
    ax.set_ylabel('Threat Contribution')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Streamlit UI
st.set_page_config(page_title="SentinelSim", layout="wide")
st.title("SentinelSim: Defense AI for Drone Threat Detection")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("Settings & History")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.25)
show_history = st.sidebar.checkbox("Show Detection History", True)

# Main content
col1, col2 = st.columns([2, 1])

# Initialize detected_objects as None
detected_objects = None

with col1:
    uploaded_file = st.file_uploader("Upload an aerial image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        try:
            # Load and process image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Load model and run detection
            model = load_model()
            results = model.predict(image, conf=confidence_threshold)
            
            # Calculate threat score
            threat_score, detected_objects = calculate_threat_score(results, image.size)
            
            # Save to history
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'threat_score': threat_score,
                'detected_objects': detected_objects
            })
            
            # Display results
            plotted_img = results[0].plot()
            st.image(plotted_img, caption="Threat Detections", use_column_width=True)
            
            # Display threat level with color coding
            threat_color = 'green' if threat_score < 30 else 'orange' if threat_score < 70 else 'red'
            st.markdown(f"### Threat Level Score: <span style='color:{threat_color}'>{threat_score:.2f}</span>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Only show the second column content if we have detected objects
if detected_objects:
    with col2:
        st.subheader("Detection Details")
        df = pd.DataFrame(detected_objects)
        st.dataframe(df)
        
        # Plot threat analysis
        fig = plot_threat_analysis(detected_objects)
        if fig:
            st.pyplot(fig)

# Show history
if show_history and st.session_state.history:
    st.sidebar.subheader("Detection History")
    for entry in reversed(st.session_state.history[-5:]):  # Show last 5 entries
        st.sidebar.markdown(f"""
        **Time**: {entry['timestamp']}  
        **Threat Score**: {entry['threat_score']:.2f}  
        **Objects Detected**: {len(entry['detected_objects'])}
        ---
        """)
