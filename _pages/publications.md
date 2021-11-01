---
layout: page
permalink: /publications/
title: publications
description: 
years: [2021, 2020, 2019, 2018]
nav: true
---
Please check out my [Google Scholar](https://scholar.google.com/citations?user=Zd7WmXUAAAAJ&hl=en) for more complete and up-to-date publication list.

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
