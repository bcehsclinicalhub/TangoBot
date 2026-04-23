document.addEventListener("DOMContentLoaded", function () {
    const btn = document.createElement("div");
    btn.innerHTML = "↑";
    btn.className = "back-to-top";
    document.body.appendChild(btn);

    window.addEventListener("scroll", () => {
        btn.style.display = window.scrollY > 200 ? "block" : "none";
    });

    btn.onclick = () => window.scrollTo({ top: 0, behavior: "smooth" });
});