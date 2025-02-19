import React, { useState } from "react";
import "./Navbar.css";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className="Navbar">
      <div className="Navbar-title">
        <Link to="/">
          <div className="Navbar-title">
            <img
              className="Navbar-title-icon"
              src="./Deepscan_icon.png"
              alt="logo"
            ></img>
          </div>
        </Link>
      </div>

      <div className="Navbar-links">
        <Link to="/deepscan">
          <button className="Navbar-scan">SCAN NOW</button>
        </Link>
      </div>
    </div>
  );
};

export default Navbar;
