var idx = lunr(function () {
  this.field('title', {boost: 10})
  this.field('excerpt')
  this.field('categories')
  this.field('tags')
  this.ref('id')
});



  
  
    idx.add({
      title: "Tuning Spark Executors Part 1",
      excerpt: "I’ve used Apache Spark at work for a couple of months and have often found the settings that control the...",
      categories: [],
      tags: [],
      id: 0
    });
    
  
    idx.add({
      title: "Tuning Spark Executors Part 2",
      excerpt: "In the previous post I discussed three of the most important settings for tuning spark executors. However, I only considered...",
      categories: [],
      tags: [],
      id: 1
    });
    
  


console.log( jQuery.type(idx) );

var store = [
  
    
    
    
      
      {
        "title": "Tuning Spark Executors Part 1",
        "url": "http://localhost:4000/spark-performance-tuing-part1/",
        "excerpt": "I’ve used Apache Spark at work for a couple of months and have often found the settings that control the...",
        "teaser":
          
            null
          
      },
    
      
      {
        "title": "Tuning Spark Executors Part 2",
        "url": "http://localhost:4000/spark-performance-tuning-part2/",
        "excerpt": "In the previous post I discussed three of the most important settings for tuning spark executors. However, I only considered...",
        "teaser":
          
            null
          
      }
    
  ]

$(document).ready(function() {
  $('input#search').on('keyup', function () {
    var resultdiv = $('#results');
    var query = $(this).val();
    var result = idx.search(query);
    resultdiv.empty();
    resultdiv.prepend('<p>'+result.length+' Result(s) found</p>');
    for (var item in result) {
      var ref = result[item].ref;
      if(store[ref].teaser){
        var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<div class="archive__item-teaser">'+
                '<img src="'+store[ref].teaser+'" alt="">'+
              '</div>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      else{
    	  var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      resultdiv.append(searchitem);
    }
  });
});
