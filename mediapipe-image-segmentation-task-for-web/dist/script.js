// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";
// Get DOM elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const webcamPredictions = document.getElementById("webcamPredictions");
const demosSection = document.getElementById("demos");
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";
let runningMode = "IMAGE";
const resultWidthHeigth = 256;
let imageSegmenter;
let labels;
const legendColors = [
    [255, 197, 0, 255],
    [128, 62, 117, 255],
    [255, 104, 0, 255],
    [166, 189, 215, 255],
    [193, 0, 32, 255],
    [206, 162, 98, 255],
    [129, 112, 102, 255],
    [0, 125, 52, 255],
    [246, 118, 142, 255],
    [0, 83, 138, 255],
    [255, 112, 92, 255],
    [83, 55, 112, 255],
    [255, 142, 0, 255],
    [179, 40, 81, 255],
    [244, 200, 0, 255],
    [127, 24, 13, 255],
    [147, 170, 0, 255],
    [89, 51, 21, 255],
    [241, 58, 19, 255],
    [35, 44, 22, 255],
    [0, 161, 194, 255] // Vivid Blue
];

// ram
function logMemoryUsage() {
    if (performance.memory) {
        console.log(`Used JS Heap: ${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        console.log(`Total JS Heap: ${(performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        console.log(`Heap Limit: ${(performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`);
    } else {
        console.warn("Performance memory API is not supported in this browser.");
    }
}
//-------------------------------------------------------------------------------------------------------
// Функция для расчёта IoU
function calculateIoU(predictedMask, groundTruthMask) {
    let intersection = 0;
    let union = 0;

    for (let i = 0; i < predictedMask.length; i++) {
        const predicted = predictedMask[i] > 0 ? 1 : 0;
        const groundTruth = groundTruthMask[i] > 0 ? 1 : 0;

        if (predicted === 1 && groundTruth === 1) {
            intersection++;
        }
        if (predicted === 1 || groundTruth === 1) {
            union++;
        }
    }

    return union === 0 ? 0 : (intersection / union);
}

// Функция для создания Ground Truth маски
function createGroundTruthMask(width, height, polygon) {
    const mask = new Uint8Array(width * height);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    // Рисуем полигон
    ctx.beginPath();
    ctx.moveTo(polygon.all_points_x[0], polygon.all_points_y[0]);
    for (let i = 1; i < polygon.all_points_x.length; i++) {
        ctx.lineTo(polygon.all_points_x[i], polygon.all_points_y[i]);
    }
    ctx.closePath();
    ctx.fill();

    // Получаем пиксели
    const imageData = ctx.getImageData(0, 0, width, height).data;
    for (let i = 0; i < imageData.length; i += 4) {
        if (imageData[i + 3] > 0) { // Альфа-канал > 0
            mask[i / 4] = 1;
        }
    }
    return mask;
}


//----------------------------------------------------------------------------------------------------------


// Настройка MediaPipe
const createImageSegmenter = async () => {
    const audio = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm");
    imageSegmenter = await ImageSegmenter.createFromOptions(audio, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
            delegate: "GPU"
        },
        runningMode: runningMode,
        outputCategoryMask: true,
        outputConfidenceMasks: false
    });
    labels = imageSegmenter.getLabels();
    demosSection.classList.remove("invisible");
};

createImageSegmenter();
const imageContainers = document.getElementsByClassName("segmentOnClick");
// Add click event listeners for the img elements.
for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i]
        .getElementsByTagName("img")[0]
        .addEventListener("click", handleClick);
}
/**
 * Demo 1: Segmented images on click and display results.
 */
let canvasClick;
async function handleClick(event) {
    // Do not segmented if imageSegmenter hasn't loaded
    if (imageSegmenter === undefined) {
        return;
    }
    canvasClick = event.target.parentElement.getElementsByTagName("canvas")[0];
    canvasClick.classList.remove("removed");
    canvasClick.width = event.target.naturalWidth;
    canvasClick.height = event.target.naturalHeight;
    const cxt = canvasClick.getContext("2d");
    cxt.clearRect(0, 0, canvasClick.width, canvasClick.height);
    cxt.drawImage(event.target, 0, 0, canvasClick.width, canvasClick.height);
    event.target.style.opacity = 0;
    // if VIDEO mode is initialized, set runningMode to IMAGE
    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await imageSegmenter.setOptions({
            runningMode: runningMode
        });
    }
    // imageSegmenter.segment() when resolved will call the callback function.
    // Измерение времени выполнения
    // Лог до сегментации
    logMemoryUsage();

    const start = performance.now();
    await imageSegmenter.segment(event.target, callback);
    const end = performance.now();
    console.log(`Latency (CPU): ${end - start} ms`);

    // Лог после сегментации
    logMemoryUsage();

    imageSegmenter.segment(event.target, callback);
}


function callback(result) {
    const cxt = canvasClick.getContext("2d");
    const { width, height } = result.categoryMask;

    // Получаем предсказанную маску
    const predictedMask = result.categoryMask.getAsUint8Array();

    // Создаём Ground Truth маску (пример полигона)

    // woman
    // const polygon = {
    //     all_points_x:[506.0203440794115,531.119945673037,546.9233985282826,554.3603175189866,557.1491621405005,552.5010877713106,540.4160944114168,525.542256430009,513.4572630701153,491.1465060980038,476.2726681165961,477.20228299043407,470.6949788735682,450.2434516491326,449.31383677529465,441.8769177845908,446.5249921537807,446.5249921537807,444.6657624061047,443.7361475322667,440.0176880369148,432.58076904621095,434.43999879388696,428.86230955085904,422.3550054339932,416.7773161909653,400.9738633357196,379.592721237446,353.56350476998256,329.39351805019504,318.23813956413926,310.8012205734354,300.57545696121764,289.42007847516186,284.7720041059719,277.3350851152681,273.61662561991614,270.8277809984022,273.61662561991614,275.4758553675921,274.5462404937541,264.32047688153636,255.02432814315654,242.00971990942477,230.85434142336902,225.27665218034113,225.27665218034113,216.91011831579928,210.4028141989334,210.4028141989334,202.96589520822957,195.52897621752572,200.17705058671564,196.4585910913637,199.24743571287763,191.8105167221738,186.2328274791459,181.584753109956,176.00706386692812,178.79590848844205,173.21821924541416,170.42937462390023,172.28860437157618,162.06284075935838,165.78130025471032,164.85168538087234,160.20361101168243,163.92207050703436,167.64053000238627,506.0203440794115],
    //     all_points_y:[466.8333333333333,456.6075697211155,444.52257636122175,425.9302788844621,411.9860557768924,391.5345285524568,372.0126162018592,343.19455511288174,326.4614873837981,283.699203187251,263.2476759628154,246.5146082337317,233.5,227.9223107569721,219.5557768924303,213.04847277556442,201.89309428950864,187.9488711819389,178.6527224435591,164.7084993359894,154.48273572377158,149.83466135458167,136.82005312084993,130.31274900398407,123.8054448871182,97.77622841965471,69.88778220451528,50.36586985391767,38.280876494023914,40.140106241699876,49.43625498007969,57.802788844621524,57.802788844621524,66.16932270916335,81.04316069057106,104.2835325365206,121.01660026560425,143.3273572377158,163.7788844621514,176.79349269588315,187.9488711819389,201.89309428950864,215.83731739707835,215.83731739707835,226.06308100929616,242.7961487383798,257.66998671978746,269.7549800796812,280.910358565737,291.1361221779548,304.15073041168654,307.8691899070385,317.1653386454183,324.60225763612215,332.039176626826,350.6314741035856,370.15338645418325,382.238379814077,384.09760956175296,390.6049136786188,400.8306772908366,408.2675962815405,418.4933598937583,425.0006640106241,432.437583001328,439.8745019920318,447.3114209827357,460.3260292164674,466.99999999999994,466.8333333333333]
    // };

    // woman bright
    // const polygon = {
    //     all_points_x:[68.06405701813878,95.07701998110174,133.26776072184248,160.28072368480542,191.0196125736943,211.51220516628692,213.37516812924986,218.03257553665728,238.52516812924986,247.83998294406467,244.11405701813877,284.16776072184246,330.74183479591653,384.7677607218424,412.7122051662869,364.27516812924983,278.57887183295355,232.936279240361,197.5399829440647,170.52701998110172,154.69183479591655,165.86961257369433,187.2936866477684,161.2122051662869,127.67887183295359,124.88442738850914,214.30664961073134,301.8659088699906,436.9307236848054,515.1751681292498,478.84739035147203,407.123316277398,307.45479775887947,289.75664961073136,0.9973903514721194,0,25.215908869990635,68.06405701813878],
    //     all_points_y:[119.22962962962961,111.77777777777776,122.95555555555553,141.58518518518517,175.1185185185185,233.8018518518518,265.4722222222222,293.41666666666663,333.47037037037035,352.09999999999997,365.1407407407407,346.51111111111106,337.1962962962962,311.1148148148148,286.89629629629627,216.10370370370367,130.4074074074074,123.88703703703702,109.9148148148148,95.01111111111109,86.62777777777777,87.55925925925925,85.69629629629628,54.02592592592592,22.355555555555554,0.9314814814814814,0.9314814814814814,74.5185185185185,199.33703703703702,288.75925925925924,335.3333333333333,395.87962962962956,453.63148148148144,500.20555555555546,500.20555555555546,182.57037037037034,137.85925925925923,119.22962962962961]
    // }

    // woman dark
    // const polygon = {
    //     all_points_x:[92.80590469925491,106.71701581036602,146.13183062518084,185.54664543999564,229.5984972918475,263.21701581036604,278.2873861807364,294.51701581036605,314.22442321777345,336.25034914369934,359.43553432888456,365.2318306251808,367.55034914369935,367.55034914369935,374.5059046992549,375.66516395851414,366.3910898844401,365.2318306251808,368.7096084029586,355.95775655110674,340.88738618073637,333.93183062518085,336.25034914369934,346.6836824770327,344.3651639585142,366.3910898844401,396.5318306251808,411.6022009955512,416.99999999999994,416.99999999999994,31.365163958514177,36.00220099555121,46.43553432888455,66.14294173629196,78.89479358814381,119.46886766221787,183.22812692147713,213.36886766221787,214.52812692147714,204.0947935881438,184.3873861807364,160.04294173629194,141.4947935881438,134.53923803258826,135.6984972918475,124.10590469925492,122.94664543999565,114.83183062518084,117.15034914369936,112.51331210666233,102.07997877332899,95.12442321777344,90.4873861807364,92.80590469925491,97.44294173629196,97.44294173629196,88.16886766221788,88.16886766221788,92.80590469925491],
    //     all_points_y:[208.66666666666666,169.25185185185185,141.42962962962963,120.56296296296296,118.24444444444444,127.51851851851852,143.74814814814815,124.04074074074073,106.65185185185184,106.65185185185184,125.19999999999999,147.2259259259259,157.65925925925924,166.93333333333334,180.84444444444443,190.1185185185185,207.50740740740738,227.21481481481482,261.9925925925926,294.4518518518518,316.47777777777776,324.59259259259255,343.14074074074074,369.80370370370366,379.0777777777778,394.14814814814815,412.6962962962963,440.5185185185185,453.27037037037036,626,626,598.1777777777778,559.9222222222222,521.6666666666666,489.2074074074074,468.3407407407407,446.3148148148148,419.65185185185186,401.1037037037037,388.35185185185185,395.30740740740737,406.9,406.9,398.7851851851852,384.87407407407403,386.0333333333333,376.75925925925924,369.80370370370366,361.68888888888887,357.05185185185184,357.05185185185184,353.5740740740741,348.93703703703704,332.7074074074074,307.2037037037037,287.4962962962963,275.9037037037037,242.28518518518518,208.66666666666666]
    // }

    // woman selfie
    // const polygon = {
    //     all_points_x:[257.4214673077619,272.1288747151693,296.12517101146557,333.28072656702113,364.24368952998407,388.23998582628036,412.23628212257665,424.62146730776186,430.81405990035444,430.0399858262804,435.4585043447989,433.91035619665075,437.00665249294707,433.1362821225767,438.5548006410952,438.5548006410952,434.6844302707248,432.3622080485026,435.4585043447989,442.4251710114656,458.6807265670211,480.3548006410952,511.31776360405814,536.0881339744285,567.8251710114656,596.4659117522064,624.332578418873,626,167.62887471516925,183.11035619665074,209.42887471516926,235.74739323368777,265.93628212257664,296.89924508553963,315.4770228633174,333.28072656702113,333.28072656702113,315.4770228633174,299.2214673077618,280.64368952998404,275.99924508553966,266.71035619665076,258.19554138183594,249.6807265670211,247.3585043447989,250.45480064109518,257.4214673077619],
    //     all_points_y:[126.17407407407408,103.72592592592594,86.69629629629631,76.63333333333334,83.60000000000001,97.53333333333335,128.4962962962963,154.81481481481484,157.91111111111113,173.39259259259262,183.45555555555558,195.0666666666667,206.6777777777778,213.64444444444447,230.6740740740741,239.962962962963,260.0888888888889,269.3777777777778,285.6333333333334,301.11481481481485,310.40370370370374,315.82222222222225,319.69259259259263,317.3703703703704,312.72592592592594,312.72592592592594,326.6592592592593,418.00000000000006,418.00000000000006,370.78148148148153,346.0111111111111,337.49629629629635,331.3037037037037,329.7555555555556,324.3370370370371,320.4666666666667,304.2111111111111,293.3740740740741,279.44074074074075,246.15555555555557,238.41481481481483,232.22222222222223,214.41851851851854,188.8740740740741,164.8777777777778,143.20370370370372,126.17407407407408]
    // }

    // woman sit
    const polygon = {
        all_points_x:[435.79775967068144,459.53109300401474,487.21998189290366,519.8533152262369,546.5533152262369,551.4977596706814,551.4977596706814,578.1977596706813,590.064426337348,614.7866485595703,635.5533152262369,648.4088707817924,650.3866485595703,649.3977596706814,655.3310930040146,669.1755374484592,680.0533152262369,704.7755374484591,711.6977596706813,706.7533152262369,750.264426337348,578.1977596706813,578.1977596706813,657.3088707817925,661.264426337348,661.264426337348,633.5755374484592,621.7088707817925,626.6533152262369,546.5533152262369,530.7310930040147,514.9088707817924,500.0755374484592,487.21998189290366,447.66442633734806,450.63109300401476,432.83109300401475,419.9755374484592,388.33109300401475,388.33109300401475,333.9422041151259,330.9755374484592,302.29775967068144,283.50887078179255,265.70887078179254,253.84220411512587,254.83109300401478,265.70887078179254,275.59775967068146,286.4755374484592,315.153315226237,329.9866485595703,359.653315226237,329.9866485595703,309.21998189290366,293.3977596706814,269.6644263373481,261.75331522623696,263.7310930040148,277.5755374484592,293.3977596706814,282.5199818929037,265.70887078179254,276.58664855957034,301.30887078179256,321.08664855957034,337.8977596706814,382.3977596706814,392.2866485595703,391.29775967068144,364.5977596706814,377.45331522623695,363.6088707817925,369.54220411512586,377.45331522623695,403.16442633734806,435.79775967068144],
        all_points_y:[14.833333333333332,11.866666666666665,20.766666666666666,42.52222222222222,91.96666666666665,144.37777777777777,175.03333333333333,193.8222222222222,209.64444444444442,230.4111111111111,280.84444444444443,315.4555555555555,340.17777777777775,366.87777777777774,383.68888888888887,401.4888888888889,427.2,441.0444444444444,460.8222222222222,476.6444444444444,533.0111111111111,532.0222222222222,526.0888888888888,521.1444444444444,511.25555555555553,502.3555555555555,499.38888888888886,498.4,486.5333333333333,486.5333333333333,497.4111111111111,497.4111111111111,492.46666666666664,488.51111111111106,474.66666666666663,463.78888888888883,456.8666666666666,475.6555555555555,474.66666666666663,400.49999999999994,404.4555555555555,465.76666666666665,458.84444444444443,440.05555555555554,423.2444444444444,403.46666666666664,386.6555555555555,379.7333333333333,365.88888888888886,365.88888888888886,386.6555555555555,401.4888888888889,399.51111111111106,382.7,361.9333333333333,352.0444444444444,354.0222222222222,344.1333333333333,341.16666666666663,322.37777777777774,312.4888888888889,309.5222222222222,308.5333333333333,300.6222222222222,296.66666666666663,309.5222222222222,338.2,370.8333333333333,358.96666666666664,340.17777777777775,294.68888888888887,258.09999999999997,208.65555555555554,158.2222222222222,62.3,27.688888888888886,14.833333333333332]
    }
    const groundTruthMask = createGroundTruthMask(width, height, polygon);

    // Рассчитываем IoU
    const iou = calculateIoU(predictedMask, groundTruthMask);
    console.log(`IoU: ${iou.toFixed(4)}`);

    // Визуализация предсказанной маски
    let imageData = cxt.getImageData(0, 0, width, height).data;
    canvasClick.width = width;
    canvasClick.height = height;
    let category = "";

    for (let i in predictedMask) {
        if (predictedMask[i] > 0) {
            category = labels[predictedMask[i]];
        }
        const legendColor = legendColors[predictedMask[i] % legendColors.length];
        imageData[i * 4] = (legendColor[0] + imageData[i * 4]) / 2;
        imageData[i * 4 + 1] = (legendColor[1] + imageData[i * 4 + 1]) / 2;
        imageData[i * 4 + 2] = (legendColor[2] + imageData[i * 4 + 2]) / 2;
        imageData[i * 4 + 3] = (legendColor[3] + imageData[i * 4 + 3]) / 2;
    }

    const uint8Array = new Uint8ClampedArray(imageData.buffer);
    const dataNew = new ImageData(uint8Array, width, height);
    cxt.putImageData(dataNew, 0, 0);

    const p = event.target.parentNode.getElementsByClassName("classification")[0];
    p.classList.remove("removed");
    p.innerText = `Category: ${category}, IoU: ${iou.toFixed(4)}`;
}



function callbackForVideo(result) {
    let imageData = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight).data;
    const mask = result.categoryMask.getAsFloat32Array();
    let j = 0;
    for (let i = 0; i < mask.length; ++i) {
        const maskVal = Math.round(mask[i] * 255.0);
        const legendColor = legendColors[maskVal % legendColors.length];
        imageData[j] = (legendColor[0] + imageData[j]) / 2;
        imageData[j + 1] = (legendColor[1] + imageData[j + 1]) / 2;
        imageData[j + 2] = (legendColor[2] + imageData[j + 2]) / 2;
        imageData[j + 3] = (legendColor[3] + imageData[j + 3]) / 2;
        j += 4;
    }
    const uint8Array = new Uint8ClampedArray(imageData.buffer);
    const dataNew = new ImageData(uint8Array, video.videoWidth, video.videoHeight);
    canvasCtx.putImageData(dataNew, 0, 0);
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
/********************************************************************
// Demo 2: Continuously grab image from webcam stream and segmented it.
********************************************************************/
// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// Get segmentation from the webcam
let lastWebcamTime = -1;
async function predictWebcam() {
    if (video.currentTime === lastWebcamTime) {
        if (webcamRunning === true) {
            window.requestAnimationFrame(predictWebcam);
        }
        return;
    }
    lastWebcamTime = video.currentTime;
    canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    // Do not segmented if imageSegmenter hasn't loaded
    if (imageSegmenter === undefined) {
        return;
    }
    // if image mode is initialized, create a new segmented with video runningMode
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await imageSegmenter.setOptions({
            runningMode: runningMode
        });
    }
    let startTimeMs = performance.now();
    // Start segmenting the stream.
    imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}
// Enable the live webcam view and start imageSegmentation.
async function enableCam(event) {
    if (imageSegmenter === undefined) {
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE SEGMENTATION";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE SEGMENTATION";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
    video.addEventListener("loadeddata", predictWebcam);
}
// If webcam supported, add event listener to button.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}