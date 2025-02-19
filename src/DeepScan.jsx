import React, { useState } from "react";
import "./DeepScan.css";
import Footer from "./Footer";

const DeepfakeDetection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedImage) {
      alert("Please select an image");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error in image prediction");
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error detecting deepfake");
      setConfidence(null);
    }
  };

  return (
    <div className="DeepScan">
      <div className="DeepScan-body">
        <h1>Deepfake Image Detection</h1>
        <div className="DeepScan-container">
          <form onSubmit={handleSubmit} className="upload-container">
            <label htmlFor="file-upload" className="upload-box">
              Click to Upload Image
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
            </label>

            {previewImage && (
              <div className="preview-container">
                <img
                  src={previewImage}
                  alt="Preview"
                  className="image-preview"
                />
              </div>
            )}
            <button type="submit" className="submit-button">
              Submit
            </button>
          </form>
        </div>

        {prediction && (
          <p className="result-text">
            Prediction: {prediction}{" "}
            {confidence !== null && `(${confidence.toFixed(2)}% real)`}
          </p>
        )}
      </div>

      <Footer />
    </div>
  );
};

export default DeepfakeDetection;
