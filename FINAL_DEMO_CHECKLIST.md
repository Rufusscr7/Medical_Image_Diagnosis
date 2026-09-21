# Final Demo Checklist

## Before Demo

- [ ] Python environment is available.
- [ ] Dependencies are installed from `requirements.txt`.
- [ ] `models/best_model.pth` is present.
- [ ] The model SHA256 is verified as `407DCFF64528FD40C97F9BA57262C46D418A010105933E4170F5F0A95DD450EA`.
- [ ] The Flask application starts without an error.
- [ ] The browser opens `http://127.0.0.1:5000/`.
- [ ] A demonstration image is available.

## During Demo

- [ ] Upload the demonstration image.
- [ ] Show the predicted class.
- [ ] Show the confidence score.
- [ ] Show the Grad-CAM overlay.
- [ ] Explain that the output is a model prediction, not a diagnosis.
- [ ] Explain that Grad-CAM visualizes regions associated with the output.
- [ ] Mention the educational and research-use disclaimer.

## Viva Preparation

- Why was HAM10000 selected?
- Why was ResNet-18 selected?
- What does preprocessing do?
- What are softmax probabilities and confidence?
- What do accuracy, precision, recall, and F1-score measure?
- What does a confusion matrix show?
- What is class imbalance, and how does it affect the results?
- What is Grad-CAM and what is its purpose?
- What are the system's limitations?
- Why is this not a clinical diagnostic system?

No new experimental results should be claimed during the demonstration. Use only the verified project results.