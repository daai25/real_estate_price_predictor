import React, { useState } from "react";
import RentalMap from "./RentalMap";

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
    <div
      style={{
        minHeight: "100vh",
        background: `
          linear-gradient(
            rgba(0, 0, 0, 0.4),
            rgba(0, 0, 0, 0.7)
          ),
          url('https://avantecture.com/wp-content/uploads/2021/10/Bruderhaus-Nr-2-aussen-13.jpg')
        `,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
        fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
        color: "white",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        padding: "2rem"
      }}
    >
      <main
        style={{
          width: "100%",
          maxWidth: "1000px",
          borderRadius: "12px",
          padding: "2rem",
        }}
      >
        {/* 🗺️ Rental Map at the top */}
        <div style={{ marginBottom: "3rem" }}>
          <h2 style={{ fontSize: "2rem", marginBottom: "1rem" }}>Rental Map</h2>
          <RentalMap />
        </div>

        {/* 📊 EDA Outputs Below */}
        <h1 style={{ fontSize: "2.5rem", marginBottom: "2rem" }}>EDA Outputs</h1>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr",
            gap: "2rem",
            justifyContent: "center",
          }}
        >
          {imageData.map((item) => (
            <div
              key={item.id}
              style={{
                padding: "1.5rem",
                borderRadius: "10px"
              }}
            >
              <h3
                style={{
                  marginBottom: "1rem",
                  fontSize: "1.3rem",
                  fontWeight: "500",
                }}
              >
                {item.title}
              </h3>
              {imageLoadError[item.id] ? (
                <div style={{ color: "tomato", fontStyle: "italic" }}>
                  Failed to load image
                </div>
              ) : (
                <img
                  src={item.path}
                  alt={item.title}
                  loading="lazy"
                  onError={() => handleImageError(item.id)}
                  style={{
                    maxWidth: "100%",
                    width: "100%",
                    height: "auto",
                    borderRadius: "6px",
                    objectFit: "contain"
                  }}
                />
              )}
            </div>
          ))}
        </div>
      </main>

      <footer
        style={{
          marginTop: "2rem",
          fontSize: "0.85rem",
          opacity: 0.7,
        }}
      >
        Michael · Josh · Enmanuel · Alessandro
      </footer>
    </div>
  );
}
