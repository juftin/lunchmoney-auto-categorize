import js from "@eslint/js";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsparser from "@typescript-eslint/parser";

export default [
  js.configs.recommended,
  {
    files: ["**/*.ts"],
    languageOptions: {
      parser: tsparser,
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: "module",
      },
      globals: {
        console: "readonly",
        document: "readonly",
        window: "readonly",
        localStorage: "readonly",
        fetch: "readonly",
        setTimeout: "readonly",
        HTMLElement: "readonly",
        HTMLButtonElement: "readonly",
        HTMLSelectElement: "readonly",
        HTMLInputElement: "readonly",
        URLSearchParams: "readonly",
        RequestInit: "readonly",
        KeyboardEvent: "readonly",
      },
    },
    plugins: {
      "@typescript-eslint": tseslint,
    },
    rules: {
      ...tseslint.configs.recommended.rules,
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "@typescript-eslint/no-explicit-any": "off",
      "no-console": "off",
    },
  },
  {
    ignores: ["node_modules/**", "dist/**", "**/*.html"],
  },
];
