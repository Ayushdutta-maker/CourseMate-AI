document.addEventListener('DOMContentLoaded', () => {
    console.log('CourseMate AI Initialized');
    
    // Smooth scroll for anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Sidebar mobility for smaller screens (can be added later)
});
