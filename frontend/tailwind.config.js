/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        quant: {
          bg: "#0B0F14",
          panel: "#121821",
          panel2: "#151C27",
          green: "#00E676",
          blue: "#2979FF",
          red: "#FF5252",
          yellow: "#FFD740",
          text: "#E6EDF3",
          muted: "#8B949E",
          line: "rgba(139,148,158,0.22)"
        }
      },
      boxShadow: {
        panel: "0 18px 60px rgba(0,0,0,0.34)"
      }
    }
  },
  plugins: []
};
