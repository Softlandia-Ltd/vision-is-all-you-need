import { createTheme, MantineColorsTuple, rem } from "@mantine/core";
import { themeToVars } from "@mantine/vanilla-extract";

// Softlandia brand colors, taken from the logo artwork.
const coral: MantineColorsTuple = [
  "#fff2ef",
  "#ffe3dd",
  "#ffc4b8",
  "#ffa390",
  "#ff876e",
  "#ff7659",
  "#ff694f",
  "#e4573f",
  "#cc4b35",
  "#b23d29",
];

const sunflower: MantineColorsTuple = [
  "#fffbe5",
  "#fff6cc",
  "#ffed99",
  "#ffe566",
  "#ffdd3d",
  "#ffd814",
  "#ffd500",
  "#e6bd00",
  "#cca800",
  "#b39200",
];

const fontFamily = "'Quicksand', sans-serif";

export const theme = createTheme({
  primaryColor: "coral",
  primaryShade: { light: 6, dark: 5 },
  defaultRadius: "md",
  fontFamily,
  // One step up from Mantine's defaults - the demo is read at arm's length.
  fontSizes: {
    xs: rem(13),
    sm: rem(15),
    md: rem(17),
    lg: rem(19),
    xl: rem(21),
  },
  lineHeights: {
    xs: "1.45",
    sm: "1.5",
    md: "1.6",
    lg: "1.6",
    xl: "1.65",
  },
  headings: { fontFamily, fontWeight: "700" },
  colors: { coral, sunflower },
  components: {
    Button: { defaultProps: { radius: "md" } },
    ActionIcon: { defaultProps: { radius: "md" } },
    Card: { defaultProps: { radius: "md" } },
    Badge: { defaultProps: { radius: "sm" } },
  },
});

export const vars = themeToVars(theme);
