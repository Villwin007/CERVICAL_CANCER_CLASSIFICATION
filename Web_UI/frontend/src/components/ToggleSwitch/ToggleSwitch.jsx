import React from "react";
import "./ToggleSwitch.css";
import { FaSun, FaMoon } from "react-icons/fa";

export default function ToggleSwitch({ theme, toggleTheme }) {
  return (
    <div className="toggle-container" aria-hidden={false}>
      <label className="switch" role="switch" aria-checked={theme === "dark"}>
        <input
          type="checkbox"
          checked={theme === "dark"}
          onChange={toggleTheme}
          aria-label="Toggle dark mode"
        />
        <span className="slider">
          <span className="icon sun"><FaSun /></span>
          <span className="icon moon"><FaMoon /></span>
        </span>
      </label>
    </div>
  );
}
