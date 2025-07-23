
import React from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/header";
import Welcome from "./components/welcome";
import Predictor from "./pages/predictor/Predictor";


function App() {
  return (
    <div>
      <Header />
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/predictor" element={<Predictor />} />
        {/* Add more routes for other pages as needed */}
      </Routes>
    </div>
  );
}

export default App;