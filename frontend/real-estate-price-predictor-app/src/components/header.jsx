
import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./header.css";

export default function Header() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="header">
      <div className="logo">realAIce</div>
      <nav className={`nav ${menuOpen ? "open" : ""}`}>
        <Link to="/">Home</Link>
        <Link to="/data_visualization">Data Visualization</Link>
        <Link to="/predictor">Predictor</Link>
        <Link to="/explorer">Interactive Explorer</Link>
        <Link to="/about">About</Link>
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