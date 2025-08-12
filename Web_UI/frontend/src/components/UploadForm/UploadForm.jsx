import React, { useState } from "react";
import { motion } from "framer-motion";
import "./UploadForm.css";

function UploadForm() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an image!");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    }

    setLoading(false);
  };

  return (
    <div className="upload-container">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="file-input"
      />

      <motion.button
        whileTap={{ scale: 0.95 }}
        whileHover={{ scale: 1.05 }}
        className="upload-btn"
        onClick={handleUpload}
      >
        {loading ? "⏳ Predicting..." : "📤 Upload & Predict"}
      </motion.button>

      {loading && (
        <motion.div
          className="loading-animation"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ repeat: Infinity, duration: 1, ease: "easeInOut" }}
        >
          <div className="dot"></div>
          <div className="dot"></div>
          <div className="dot"></div>
        </motion.div>
      )}

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="result-card"
        >
          <h3>Prediction: {result.class}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>

          <div className="confidence-bar">
            <motion.div
              className="confidence-fill"
              initial={{ width: 0 }}
              animate={{ width: `${result.confidence * 100}%` }}
              transition={{ duration: 0.8 }}
            />
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default UploadForm;
