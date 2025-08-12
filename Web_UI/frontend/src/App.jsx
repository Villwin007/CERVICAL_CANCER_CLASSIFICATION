import React, { useState, useEffect } from "react";
import ToggleSwitch from "./components/ToggleSwitch/ToggleSwitch";
import UploadForm from "./components/UploadForm/UploadForm";
import "./App.css";
import { motion } from "framer-motion";

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <div className={`app-root ${theme}`}>
      {/* Top-left theme toggle */}
      <ToggleSwitch theme={theme} toggleTheme={toggleTheme} />

      {/* Main content */}
      <main className="app-container">
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="header"
        >
          <h1>🩺 Cervical Cancer Detection</h1>
          <p className="subtitle">Upload an image to get prediction using the DARTS model </p>
        </motion.div>

        <UploadForm theme={theme} />
      </main>
    </div>
  );
}
