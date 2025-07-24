import React from "react";
import { useNavigate } from "react-router-dom";

export default function Welcome() {
  const navigate = useNavigate();

  return (
    <div
      style={{
        minHeight: "100vh",
        background: `
          linear-gradient(
            rgba(0, 0, 0, 0.4),
            rgba(0, 0, 0, 0.7)
          ),
          url('https://avantecture.com/wp-content/uploads/2021/10/Bruderhaus-Nr-2-aussen-13.jpg') center/cover no-repeat
        `,
        fontFamily: '"Helvetica Neue", Helvetica, Arial, sans-serif',
        color: "white",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        padding: "0 2rem",
        position: "relative",
      }}
    >
      <main
        style={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          width: "100%",
          paddingTop: "4rem",
        }}
      >
        <h1
          style={{
            fontSize: "3rem", // vorher 4rem
            fontWeight: "600",
            marginBottom: "1rem",
            letterSpacing: "-0.5px",
            lineHeight: "1.2",
            maxWidth: "90%",
          }}
        >
          Empower Your Real Estate Decisions
        </h1>
        <p
          style={{
            fontSize: "1.2rem", // vorher 1.5rem
            marginBottom: "2rem",
            opacity: 0.85,
            maxWidth: "600px",
          }}
        >
          Predict property prices with confidence. Data-driven. Elegant. Instant.
        </p>
        <div>
          <button
            onClick={() => navigate('/predictor')}
            style={{
              padding: "0.6rem 1.5rem",
              fontSize: "0.95rem",
              fontWeight: "500",
              color: "#000",
              backgroundColor: "white",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
              boxShadow: "0 2px 4px rgba(0,0,0,0.3)",
              marginRight: "1rem",
            }}
          >
            Get Started
          </button>
          <button
            onClick={() => navigate('/about')}
            style={{
              padding: "0.6rem 1.5rem",
              fontSize: "0.95rem",
              fontWeight: "500",
              color: "white",
              backgroundColor: "transparent",
              border: "1px solid white",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            Learn More
          </button>
        </div>
      </main>

      <footer
        style={{
          textAlign: "center",
          padding: "1rem",
          fontSize: "0.85rem",
          opacity: 0.7,
        }}
      >
        Michael &middot; Josh &middot; Enmanuel &middot; Alessandro
      </footer>
    </div>
  );
}