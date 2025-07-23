import React, { useState } from "react";
import "./ImageTablePage.css"; // styling for this page
import RentalMap from "./RentalMap"; // adjust this import path if needed

const imageData = [
  {
    id: 1,
    title: "Confusion Matrix",
    path: "/graphs/polished_correlation_heatmap.png"
  },
  {
    id: 2,
    title: "Average Price by Region",
    path: "/graphs/gradient_avg_price_by_region.png"
  }
];

export default function ImageTablePage() {
  const [imageLoadError, setImageLoadError] = useState({});

  const handleImageError = (id) => {
    setImageLoadError((prev) => ({
      ...prev,
      [id]: true
    }));
  };

  return (
    <div className="image-table-container">
      {/* 🗺️ Rental Map at the top */}
      <div style={{ marginBottom: "40px" }}>
        <h2 style={{ textAlign: "center", marginBottom: "12px" }}>Rental Map</h2>
        <RentalMap />
      </div>

      {/* 📊 EDA Outputs Below */}
      <h1>EDA Outputs</h1>
      <div className="image-grid">
        {imageData.map((item) => (
          <div key={item.id} className="image-item">
            <div className="image-title">{item.title}</div>
            {imageLoadError[item.id] ? (
              <div className="image-error">Failed to load image</div>
            ) : (
              <img
                src={item.path}
                alt={item.title}
                className="visualization-image"
                loading="lazy"
                onError={() => handleImageError(item.id)}
                style={{
                  display: "block",
                  margin: "0 auto",
                  maxWidth: "100%",
                  height: "auto",
                  width: "1000px"
                }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}