import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./Navbar";
import "./App.css";
import Homepage from "./Homepage";
import DeepScan from "./DeepScan";

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="App-container">
          <Routes>
            <Route path="/" element={<Homepage />} />
            <Route path="/deepscan" element={<DeepScan />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
