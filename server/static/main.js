document.addEventListener('DOMContentLoaded', function() {
    const API_BASE = 'http://localhost:5000/api';

    const categorySelect = document.getElementById('categorySelect');
    const searchQuery = document.getElementById('searchQuery');
    const searchButton = document.getElementById('searchButton');
    const recommendations = document.getElementById('recommendations');
    const randomArticles = document.getElementById('randomArticles');

    loadCategories();
    loadRandomArticles();

    searchButton.addEventListener('click', handleSearch);
    searchQuery.addEventListener('keypress', e => {
        if (e.key === 'Enter') handleSearch();
    });

    async function loadCategories() {
        try {
            const res = await fetch(`${API_BASE}/categories`);
            const data = await res.json();
            data.categories.forEach(cat => {
                const opt = document.createElement('option');
                opt.value = cat.name;
                opt.textContent = `${cat.name} (${cat.article_count})`;
                categorySelect.appendChild(opt);
            });
        } catch (err) {
            console.error('Failed to load categories:', err);
        }
    }

    async function handleSearch() {
        const query = searchQuery.value.trim();
        if (!query) return alert('Please enter a search query');

        try {
            const res = await fetch(`${API_BASE}/recommend`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query, top_k: 5 })
            });
            const articles = await res.json();
            displayRecommendations(articles);
        } catch (err) {
            console.error(err);
        }
    }

    async function loadRandomArticles() {
        try {
            const res = await fetch(`${API_BASE}/random_articles?n=6`);
            const articles = await res.json();
            displayRandomArticles(articles);
        } catch (err) {
            console.error(err);
        }
    }

    function displayRecommendations(articles) {
        recommendations.innerHTML = articles.map(a => `
            <div class="recommendation">
                <h5>${a.headline}</h5>
                <div class="text-muted">${a.category}</div>
                <p>${a.short_description}</p>
            </div>
        `).join('');
    }

    function displayRandomArticles(articles) {
        randomArticles.innerHTML = articles.map(a => `
            <div class="col-md-4 mb-3">
                <div class="card h-100">
                    <div class="card-body">
                        <h6 class="card-title">${a.headline}</h6>
                        <p class="text-muted">${a.category}</p>
                    </div>
                </div>
            </div>
        `).join('');
    }
});
