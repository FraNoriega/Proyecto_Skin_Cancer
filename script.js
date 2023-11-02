var canvas = document.getElementById("canvas");
var video = document.getElementById("video");
var ctx = canvas.getContext("2d");
var defaultDisplay = document.getElementById("default")
var infoDiv = document.getElementById("infoDiv")
var text = document.getElementById("message")
var input = document.getElementById('imagenInput')
var modelo = null;
var othercanvas = document.getElementById('othercanvas')
var size = 400;
var camaras = [];
var currentStream = null;
const clases = ["Enfermedad de bowen", "Carcinoma de células basales", "Queratosis",
"Dermatofibroma", "Melanoma", "Nevus melanocíticos", "Lesiones vasculares"];
const mensajes = ["La enfermedad de Bowen es una forma de carcinoma espinocelular in situ que afecta a todo el grosor de la epidermis e invade las unidades pilosebáceas. Aunque suele permanecer in situ durante largos períodos de tiempo, del 3% al 5% de los pacientes no tratados pueden desarrollar un carcinoma invasivo.",
"Las células basales producen nuevas células de la piel a medida que las anteriores mueren. Limitar la exposición al sol puede prevenir que estas células se tornen cancerosas. Este tipo de cáncer generalmente se manifiesta como una protuberancia cerosa blanquecina o un área escamosa amarronada en las zonas que se exponen al sol, como el rostro y el cuello.El tratamiento incluye la prescripción de cremas o la cirugía para extirpar el cáncer.",
"Una queratosis actínica es una mancha áspera y escamosa en la piel que se presenta después de años de exposición al sol. A menudo aparece en la cara, los labios, las orejas, los antebrazos, el cuero cabelludo, el cuello o el dorso de las manos.",
"El dermatofibroma es una lesión benigna de estirpe fibrohistiocitaria muy frecuente que suele aparecer en adultos jóvenes con predominio en el sexo femenino. La forma clínica más frecuente es la de un nódulo solitario asintomático de pocos milímetros y color parduzco con predilección por los miembros inferiores.",
"El melanoma ocurre cuando las células productoras de pigmento que dan color a la piel se vuelven cancerosas. Los síntomas incluyen neoplasias inusuales y nuevas, o cambios en un lunar ya existente. Los melanomas pueden aparecer en cualquier lugar del cuerpo. El tratamiento puede incluir cirugía, radioterapia, medicamentos y, en algunos casos, quimioterapia.",
"Trastorno generalmente benigno de las células de la piel productoras de pigmento, comúnmente llamado marca de nacimiento o lunar. Este tipo de lunar suele ser de gran tamaño y está ocasionado por un trastorno de los melanocitos, las células que producen el pigmento (melanina). Los nevos melanocíticos pueden ser ásperos, planos o elevados. Pueden estar presentes en el momento del nacimiento o aparecer más adelante. Aunque es raro que ocurra, los nevos melanocíticos pueden ser cancerosos. La mayoría de los casos no requieren tratamiento, aunque a veces es necesario extirpar el lunar.",
"Un traumatismo vascular es una lesión de una arteria o vena como consecuencia de un traumatismo o golpe. Pueden afectar al sistema arterial, linfático o venoso, y suelen ocurrir con mayor frecuencia en las extremidades, sobre todo en las inferiores (en el 80-90% de los casos)."];

(async () => {
    console.log("Cargando modelo...");
    modelo = await tf.loadLayersModel("model.json");
    console.log("Modelo cargado...");
})();

/*window.onload = function() {
    mostrarCamara();
}*/
function cargar_imagen(){
    console.log('a')
    var ctx = canvas.getContext("2d");

    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = function (e) {
        const imagen = new Image();
        imagen.src = e.target.result;

        imagen.onload = function () {
            imagen.width = 400
            imagen.height = 400
            canvas.width = imagen.width;
            canvas.height = imagen.height;
            ctx.drawImage(imagen, 0, 0, imagen.width, imagen.height);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixeles = imageData.data;

            // Ahora, 'pixeles' contiene un array con los datos de píxeles de la imagen
            console.log(pixeles);
        };
    }
    reader.readAsDataURL(file);
    setTimeout(procesarCamara, 20);
    

    
}

function mostrarCamara() {

    var opciones = {
        audio: false,
        video: {
            facingMode: "user", width: size, height: size
        }
    };

    if(navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia(opciones)
            .then(function(stream) {
                currentStream = stream;
                video.srcObject = currentStream;
                procesarCamara();


            })
            .catch(function(err) {
                alert("No se pudo utilizar la camara :(");
                console.log("No se pudo utilizar la camara :(", err);
                alert(err);
            })
    } else {
        alert("No existe la funcion getUserMedia... oops :( no se puede usar la camara");
    }
}

function predecir() {
    if (modelo != null) {
        cargar_imagen()
        resample_single(canvas, 100, 100, othercanvas);

        var ctx2 = othercanvas.getContext("2d");

        var imgData = ctx2.getImageData(0,0,100,100);
        var arr = []; 
        var arr100 = []; 
        for (var p=0, i=0; p < imgData.data.length; p+=4) {
            var red = imgData.data[p]/255;
            var green = imgData.data[p+1]/255;
            var blue = imgData.data[p+2]/255;
            arr100.push([red, green, blue]); 
            if (arr100.length == 100) {
                arr.push(arr100);
                arr100 = [];
            }
        }

        arr = [arr]; 
        var tensor4 = tf.tensor4d(arr);
        var resultados = modelo.predict(tensor4).dataSync();
        var mayorIndice = resultados.indexOf(Math.max.apply(null, resultados));
        defaultDisplay.style.display = 'none'
        infoDiv.style.display = 'inline-block'
        text.innerHTML = `Se ha detectado <b>${clases[mayorIndice]}</b> con un porcentaje de %${parseFloat((resultados[mayorIndice]*100).toFixed(2))} de seguridad
        <br><br>${mensajes[mayorIndice]}`

    }

    
}

function procesarCamara() {
          
    var ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, size, size, 0, 0, size, size);

    setTimeout(procesarCamara, 20);
}

function resample_single(canvas, width, height, resize_canvas) {
    var width_source = canvas.width;
    var height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    var ratio_w = width_source / width;
    var ratio_h = height_source / height;
    var ratio_w_half = Math.ceil(ratio_w / 2);
    var ratio_h_half = Math.ceil(ratio_h / 2);

    var ctx = canvas.getContext("2d");
    var ctx2 = resize_canvas.getContext("2d");
    var img = ctx.getImageData(0, 0, width_source, height_source);
    var img2 = ctx2.createImageData(width, height);
    var data = img.data;
    var data2 = img2.data;

    for (var j = 0; j < height; j++) {
        for (var i = 0; i < width; i++) {
            var x2 = (i + j * width) * 4;
            var weight = 0;
            var weights = 0;
            var weights_alpha = 0;
            var gx_r = 0;
            var gx_g = 0;
            var gx_b = 0;
            var gx_a = 0;
            var center_y = (j + 0.5) * ratio_h;
            var yy_start = Math.floor(j * ratio_h);
            var yy_stop = Math.ceil((j + 1) * ratio_h);
            for (var yy = yy_start; yy < yy_stop; yy++) {
                var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                var center_x = (i + 0.5) * ratio_w;
                var w0 = dy * dy; //pre-calc part of w
                var xx_start = Math.floor(i * ratio_w);
                var xx_stop = Math.ceil((i + 1) * ratio_w);
                for (var xx = xx_start; xx < xx_stop; xx++) {
                    var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                    var w = Math.sqrt(w0 + dx * dx);
                    if (w >= 1) {
                        //pixel too far
                        continue;
                    }
                    //hermite filter
                    weight = 2 * w * w * w - 3 * w * w + 1;
                    var pos_x = 4 * (xx + yy * width_source);
                    //alpha
                    gx_a += weight * data[pos_x + 3];
                    weights_alpha += weight;
                    //colors
                    if (data[pos_x + 3] < 255)
                        weight = weight * data[pos_x + 3] / 250;
                    gx_r += weight * data[pos_x];
                    gx_g += weight * data[pos_x + 1];
                    gx_b += weight * data[pos_x + 2];
                    weights += weight;
                }
            }
            data2[x2] = gx_r / weights;
            data2[x2 + 1] = gx_g / weights;
            data2[x2 + 2] = gx_b / weights;
            data2[x2 + 3] = gx_a / weights_alpha;
        }
    }


    ctx2.putImageData(img2, 0, 0);
}