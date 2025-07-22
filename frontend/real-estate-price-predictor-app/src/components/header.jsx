import React, { useState } from "react";
import "./header.css";

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="header">
      <div className="logo">realAIce</div>
      <nav className={`nav ${menuOpen ? "open" : ""}`}>
        <a href="#home">Home</a>
        <a href="#visualization">Data Visualization</a>
        <a href="#predictor">Predictor</a>
        <a href="#explorer">Interactive Explorer</a>
        <a href="#about">About</a>
      </nav>
      <div
        className={`hamburger ${menuOpen ? "open" : ""}`}
        onClick={() => setMenuOpen(!menuOpen)}
      >
        <span />
        <span />
        <span />
      </div>
    </header>
  );
}