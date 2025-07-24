import React from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/header";
import Welcome from "./components/welcome";
import Predictor from "./pages/predictor/Predictor";
import DataVisualization from "./pages/data_visualization/DataVisualization";
import AboutPage from "./pages/about/About";import Explorer from "./pages/explorer/Explorer";

function App() {
  return (
    <div>
      <Header />
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/predictor" element={<Predictor />} />
        <Route path="/explorer" element={<Explorer />} />
        <Route path="/data_visualization" element={<DataVisualization />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </div>
  );
}

export default App;