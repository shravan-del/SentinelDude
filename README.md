# 🛡️ SentinelSim: AI-Powered Drone Threat Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentinelsim.streamlit.app)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="demo/header.gif" alt="SentinelSim Demo" width="800"/>
</p>

## 🎯 Overview

SentinelSim is a sophisticated AI-powered defense system that demonstrates advanced object tracking and threat prediction capabilities. Built with modern machine learning technologies, it showcases the integration of computer vision with real-world defense applications.

### 🌟 Key Features

- **Real-time Threat Detection**: Utilizes YOLOv8 for accurate object detection
- **Advanced Threat Scoring**: Intelligent algorithm considering object size, position, and confidence
- **Interactive Dashboard**: Real-time visualization of threat levels and detection metrics
- **Historical Analysis**: Track and analyze detection patterns over time
- **Responsive Design**: Seamless experience across devices

## 🛠️ Technology Stack

- **Backend**: Python, PyTorch, OpenCV
- **ML Model**: YOLOv8 (State-of-the-art object detection)
- **Frontend**: Streamlit (Interactive UI)
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Plotly

## 🚀 Live Demo

Try the live demo: [SentinelSim Demo](https://sentinelsim.streamlit.app)

## 📊 Sample Results

<p align="center">
  <img src="demo/detection.png" alt="Threat Detection" width="400"/>
  <img src="demo/analysis.png" alt="Threat Analysis" width="400"/>
</p>

## 💻 Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SentinelSim.git
cd SentinelSim
```

2. Create and activate virtual environment:
```bash
python -m venv sentinelsim-env
source sentinelsim-env/bin/activate  # On Windows: sentinelsim-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 📱 Deployment

The app is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Sign up on [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your forked repository
4. Deploy!

## 🔒 Security Considerations

This is a simulation project intended for educational and demonstration purposes. For real-world applications, additional security measures would be necessary:

- Secure data transmission
- Access control and authentication
- Model security and validation
- Privacy considerations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Connect

- [LinkedIn](https://linkedin.com/in/yourusername)
- [Portfolio](https://yourportfolio.com)
- [GitHub](https://github.com/yourusername)

---

<p align="center">
  Made with ❤️ for the defense tech community
</p> 