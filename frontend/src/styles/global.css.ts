import { globalStyle } from "@vanilla-extract/css";

// Default style resets

globalStyle("html, body", {
  margin: 0,
  padding: 0,
  minHeight: "100vh",
  height: "100%",
  fontFamily: "'Quicksand', sans-serif",
});

globalStyle("h1, h2, h3, h4, h5, h6, p, a, span, label, div", {
  fontWeight: "inherit",
  fontSize: "inherit",
  color: "inherit",
  textDecoration: "none",
  margin: 0,
  padding: 0,
  fontFamily: "'Quicksand', sans-serif",
});

globalStyle("button", {
  WebkitTouchCallout: "none", // iOS Safari
  WebkitUserSelect: "none", // Safari
  MozUserSelect: "none", // Old versions of Firefox
  msUserSelect: "none", // Internet Explorer / Edge
  userSelect: "none", // Chrome, Edge, Opera and Firefox
});

// Root styles

globalStyle("#root", {
  height: "100%",
  fontFamily: "'Quicksand', sans-serif",
});
