/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        quant: {
          bg: "#0C1117",
          panel: "#111820",
          panel2: "#151D27",
          green: "#14B8A6",
          blue: "#4F8EF7",
          red: "#F87171",
          yellow: "#FBBF24",
          text: "#E8EEF6",
          muted: "#9AA6B2",
          line: "rgba(154,166,178,0.22)"
        }
      },
      boxShadow: {
        panel: "0 12px 34px rgba(0,0,0,0.24)"
      }
    }
  },
  plugins: []
};
