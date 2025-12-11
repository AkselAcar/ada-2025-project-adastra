const sections = document.querySelectorAll("section");
const navLinks = document.querySelectorAll(".navbar a");

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;

      navLinks.forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("href") === "#" + id) {
          link.classList.add("active");
        }
      });
    }
  });
}, {
  threshold: 0.5  // Section must be 50% visible → more accurate highlighting
});

sections.forEach(section => observer.observe(section));
