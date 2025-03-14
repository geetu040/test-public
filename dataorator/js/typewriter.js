// js/typewriter.js

document.addEventListener("DOMContentLoaded", function () {
  const heading = document.querySelector(".main-heading");
  const text = "CHAT WITH YOUR DATABASE";
  let index = 0;
  heading.textContent = ""; // Ensure heading is empty

  function typeWriter() {
    if (index < text.length) {
      heading.textContent += text.charAt(index);
      index++;
      setTimeout(typeWriter, 100); // Adjust delay (in ms) for typing speed
    }
  }

  typeWriter();
});
