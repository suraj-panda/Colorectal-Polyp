/*window.setTimeout(function(){
    $(".alert").fadeTo(500,0).slideUP(500, function(){
        $(this).remove();
    });
}, 3000);*/

if(window.hustory.replaceState){
    window.history.replaceState(null,null,window.location.href);
}
