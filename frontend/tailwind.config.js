/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        quant: {
          bg: "#EEF2F0",
          panel: "rgba(255,255,255,0.64)",
          panel2: "rgba(255,255,255,0.48)",
          green: "#1F7A5C",
          blue: "#2F5F8F",
          red: "#B84A45",
          yellow: "#9A6B24",
          text: "#17211D",
          muted: "#66746E",
          line: "rgba(35,48,42,0.14)"
        }
      },
      boxShadow: {
        panel: "0 22px 80px rgba(31,42,37,0.10)"
      }
    }
  },
  plugins: []
};
