styleNames = ["Art Deco", "Art Nouveau", "Baroque", "Bauhaus & International", "Brutalist", "Byzantine", "Chicago School", "Chinese", "Deconstructivism", "Gothic", "Modernism", "Neoclassical & Beaux-Arts", "Palladian", "Postmodernism", "Romanesque", "Russian Revival"];
styleName = "";

window.addEventListener('load', function() {
    document.querySelector('input[type="file"]').addEventListener('change', function() {
        if (this.files && this.files[0]) {
            var img = document.querySelector("#inputPreview");
            img.onload = () => {
                URL.revokeObjectURL(img.src); // no longer needed, free memory
            }
            img.src = URL.createObjectURL(this.files[0]);
            var file = document.getElementById("inputImage");
            $.ajax({
                url: "/predict",
                type: "GET",
                data: {
                    fileName: file.value
                },
                success: function(result) {
                    var txt = document.querySelector("#styleName");
                    txt.innerHTML = "This is a " + result + " architecture!";
                }
            });
        }
    });
});

function transferStyle(styleIndex) {
    var img = document.querySelector("#resultPreview");
    img.onload = () => {
        URL.revokeObjectURL(img.src); // no longer needed, free memory
    }
    img.src = "/static/image/Programming/StyleDemo/Results/" + styleNames[styleIndex] + ".jpg";
}

function validateMyForm() {
    return false;
}