var idx = lunr(function () {
  this.field('title', {boost: 10})
  this.field('excerpt')
  this.field('categories')
  this.field('tags')
  this.ref('id')
});



  
  
    idx.add({
      title: "Test Post 1",
      excerpt: "This is a test post.\n",
      categories: [],
      tags: ["r","career","statistics","education"],
      id: 0
    });
    
  
    idx.add({
      title: "Estimating Trump's Fake Twitter Followers: Part 1",
      excerpt: "Donald Trump enjoys using his Twitter account to “go around” the “Fake News Media” and has amassed a large following...",
      categories: [],
      tags: [],
      id: 1
    });
    
  
    idx.add({
      title: "Estimating Trump's Fake Twitter Followers: Part 2",
      excerpt: "In the previous post I collected some data on a sample of Donald Trump’s Twitter followers and also assembled a...",
      categories: [],
      tags: [],
      id: 2
    });
    
  


console.log( jQuery.type(idx) );

var store = [
  
    
    
    
      
      {
        "title": "Test Post 1",
        "url": "http://localhost:4000/test-post1/",
        "excerpt": "This is a test post.\n",
        "teaser":
          
            null
          
      },
    
      
      {
        "title": "Estimating Trump's Fake Twitter Followers: Part 1",
        "url": "http://localhost:4000/trump_fakeaccounts_part1/",
        "excerpt": "Donald Trump enjoys using his Twitter account to “go around” the “Fake News Media” and has amassed a large following...",
        "teaser":
          
            null
          
      },
    
      
      {
        "title": "Estimating Trump's Fake Twitter Followers: Part 2",
        "url": "http://localhost:4000/trump_fakeaccounts_part2/",
        "excerpt": "In the previous post I collected some data on a sample of Donald Trump’s Twitter followers and also assembled a...",
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
