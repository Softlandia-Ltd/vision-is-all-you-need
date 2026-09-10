import { globalStyle } from "@vanilla-extract/css";

// Default style resets. Typography itself is owned by the Mantine theme,
// so we only strip browser defaults here.

globalStyle("html, body", {
  margin: 0,
  padding: 0,
  minHeight: "100vh",
  height: "100%",
});

globalStyle("h1, h2, h3, h4, h5, h6, p", {
  margin: 0,
});

globalStyle("a", {
  color: "inherit",
  textDecoration: "none",
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
});
