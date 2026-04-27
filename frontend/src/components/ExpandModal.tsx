import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

type ExpandModalProps = {
  onClose: () => void;
  children: React.ReactNode;
};

export function ExpandModal({ onClose, children }: ExpandModalProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setVisible(true));

    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") handleClose();
    }
    window.addEventListener("keydown", onKey);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  function handleClose() {
    setVisible(false);
    setTimeout(onClose, 220);
  }

  return createPortal(
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center p-8 transition-opacity duration-200 ${
        visible ? "opacity-100" : "opacity-0"
      }`}
    >
      <div
        className={`absolute inset-0 bg-black/70 transition-all duration-200 ${
          visible ? "backdrop-blur-sm" : "backdrop-blur-none"
        }`}
        onClick={handleClose}
      />

      <div
        className={`scrollbar-quant relative z-10 max-h-[90vh] w-full max-w-5xl overflow-auto transition-all duration-200 ${
          visible ? "scale-100 opacity-100" : "scale-95 opacity-0"
        }`}
      >
        <button
          onClick={handleClose}
          className="absolute right-3 top-3 z-20 rounded-md border border-quant-line bg-quant-panel2/90 p-1.5 text-quant-muted backdrop-blur-sm transition hover:text-quant-text"
        >
          <X size={16} />
        </button>
        {children}
      </div>
    </div>,
    document.body,
  );
}
