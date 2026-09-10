import { style } from "@vanilla-extract/css";
import { vars } from "../theme";
import { rem } from "@mantine/core";

export const dropzone = style({
  width: "100%",
  padding: rem(12),
  border: `1px dashed ${vars.colors.gray[4]}`,
  borderRadius: vars.radius.md,
  backgroundColor: "transparent",
  transition: "background-color 150ms ease, border-color 150ms ease",
  selectors: {
    "&:hover": {
      backgroundColor: vars.colors.gray[0],
      borderColor: vars.colors.coral[6],
    },
    [vars.darkSelector]: {
      borderColor: vars.colors.dark[4],
    },
    [`${vars.darkSelector}:hover`]: {
      backgroundColor: vars.colors.dark[6],
      borderColor: vars.colors.coral[5],
    },
  },
});

export const dropzoneLg = style({
  padding: rem(28),
});
