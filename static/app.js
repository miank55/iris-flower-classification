(() => {
  const fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
  const buttons = document.querySelectorAll("[data-values]");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const values = (button.dataset.values || "").split(",");
      if (values.length !== fields.length) return;
      fields.forEach((name, index) => {
        const input = document.getElementById(name);
        if (input) input.value = values[index];
      });
      const first = document.getElementById(fields[0]);
      if (first) first.focus({ preventScroll: true });
      const form = document.querySelector(".prediction-form");
      if (form) form.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
})();