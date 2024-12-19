/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import "bulma/css/bulma.css";
import "@tensorflow/tfjs-backend-webgl";

import { load } from "@tensorflow-models/deeplab";
import * as tf from "@tensorflow/tfjs-core";

import ade20kExampleImage from "./examples/ade20k.jpg";
import cityscapesExampleImage from "./examples/cityscapes.jpg";
import pascalExampleImage from "./examples/pascal.jpg";

const modelNames = ["pascal", "cityscapes", "ade20k"];
const deeplab = {};
const state = {};

const deeplabExampleImages = {
  pascal: pascalExampleImage,
  cityscapes: cityscapesExampleImage,
  ade20k: ade20kExampleImage,
};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle("is-invisible", force);
};

const initializeModels = async () => {
  modelNames.forEach((base) => {
    const selector = document.getElementById("quantizationBytes");
    const quantizationBytes = Number(
        selector.options[selector.selectedIndex].text,
    );
    state.quantizationBytes = quantizationBytes;
    deeplab[base] = load({ base, quantizationBytes });
    const toggler = document.getElementById(`toggle-${base}-image`);
    toggler.onclick = () => setImage(deeplabExampleImages[base]);
    const runner = document.getElementById(`run-${base}`);
    runner.onclick = async () => {
      toggleInvisible("output-card", true);
      toggleInvisible("legend-card", true);
      await tf.nextFrame();
      await runDeeplab(base);
    };
  });
  const uploader = document.getElementById("upload-image");
  uploader.addEventListener("change", processImages);
  status("Initialised models, waiting for input...");
};

const setImage = (src) => {
  toggleInvisible("output-card", true);
  toggleInvisible("legend-card", true);
  const image = document.getElementById("input-image");
  image.src = src;
  toggleInvisible("input-card", false);
  status("Waiting until the model is picked...");
};

const processImage = (file) => {
  if (!file.type.match("image.*")) {
    return;
  }
  const reader = new FileReader();
  reader.onload = (event) => {
    setImage(event.target.result);
  };
  reader.readAsDataURL(file);
};

const processImages = (event) => {
  const files = event.target.files;
  Array.from(files).forEach(processImage);
};

// const displaySegmentationMap = (modelName, deeplabOutput) => {
//   const { legend, height, width, segmentationMap } = deeplabOutput;
//   const canvas = document.getElementById("output-image");
//   const ctx = canvas.getContext("2d");

//   toggleInvisible("output-card", false);
//   const segmentationMapData = new ImageData(segmentationMap, width, height);
//   canvas.style.width = "100%";
//   canvas.style.height = "100%";
//   canvas.width = width;
//   canvas.height = height;
//   ctx.putImageData(segmentationMapData, 0, 0);

//   const legendList = document.getElementById("legend");
//   while (legendList.firstChild) {
//     legendList.removeChild(legendList.firstChild);
//   }

//   Object.keys(legend).forEach((label) => {
//     const tag = document.createElement("span");
//     tag.innerHTML = label;
//     const [red, green, blue] = legend[label];
//     tag.classList.add("column");
//     tag.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
//     tag.style.padding = "1em";
//     tag.style.margin = "1em";
//     tag.style.color = "#ffffff";

//     legendList.appendChild(tag);
//   });
//   toggleInvisible("legend-card", false);

//   const inputContainer = document.getElementById("input-card");
//   inputContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
// };

const calculateIoU = (segmentationMap, groundTruthPolygon, width, height) => {
  // Создаём пустое изображение для истинных меток
  const groundTruthMask = new Uint8Array(width * height);
  // ----------------------------
  console.log(`SegmentationMap size: ${segmentationMap.length}`);
  console.log(`Width * Height: ${width * height}`);

  // Рендерим полигон истинных меток в битовую карту
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  ctx.beginPath();
  ctx.moveTo(groundTruthPolygon.all_points_x[0], groundTruthPolygon.all_points_y[0]);
  for (let i = 1; i < groundTruthPolygon.all_points_x.length; i++) {
      ctx.lineTo(groundTruthPolygon.all_points_x[i], groundTruthPolygon.all_points_y[i]);
  }
  ctx.closePath();
  ctx.fillStyle = "white";
  ctx.fill();

  const imageData = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < imageData.data.length; i += 4) {
      // Если пиксель закрашен (белый), то включаем его в истинные метки
      groundTruthMask[i / 4] = imageData.data[i] > 0 ? 1 : 0;
  }

  // Подсчитываем пересечение и объединение
  let intersection = 0;
  let union = 0;

  for (let i = 0; i < segmentationMap.length; i++) {
      const pred = segmentationMap[i] > 0 ? 1 : 0; // Бинаризуем предсказания
      const truth = groundTruthMask[i];

      if (pred === 1 && truth === 1) {
          intersection++;
      }
      if (pred === 1 || truth === 1) {
          union++;
      }
  }

  return 1 - (union === 0 ? 0 : intersection / union); // Избегаем деления на 0
};

const displaySegmentationMap = (modelName, deeplabOutput) => {
  const { legend, height, width, segmentationMap } = deeplabOutput;
  const canvas = document.getElementById("output-image");
  const ctx = canvas.getContext("2d");

  toggleInvisible("output-card", false);
  const segmentationMapData = new ImageData(segmentationMap, width, height);
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.width = width;
  canvas.height = height;
  ctx.putImageData(segmentationMapData, 0, 0);
  const polygon = {
    all_points_x:[506.0203440794115,531.119945673037,546.9233985282826,554.3603175189866,557.1491621405005,552.5010877713106,540.4160944114168,525.542256430009,513.4572630701153,491.1465060980038,476.2726681165961,477.20228299043407,470.6949788735682,450.2434516491326,449.31383677529465,441.8769177845908,446.5249921537807,446.5249921537807,444.6657624061047,443.7361475322667,440.0176880369148,432.58076904621095,434.43999879388696,428.86230955085904,422.3550054339932,416.7773161909653,400.9738633357196,379.592721237446,353.56350476998256,329.39351805019504,318.23813956413926,310.8012205734354,300.57545696121764,289.42007847516186,284.7720041059719,277.3350851152681,273.61662561991614,270.8277809984022,273.61662561991614,275.4758553675921,274.5462404937541,264.32047688153636,255.02432814315654,242.00971990942477,230.85434142336902,225.27665218034113,225.27665218034113,216.91011831579928,210.4028141989334,210.4028141989334,202.96589520822957,195.52897621752572,200.17705058671564,196.4585910913637,199.24743571287763,191.8105167221738,186.2328274791459,181.584753109956,176.00706386692812,178.79590848844205,173.21821924541416,170.42937462390023,172.28860437157618,162.06284075935838,165.78130025471032,164.85168538087234,160.20361101168243,163.92207050703436,167.64053000238627,506.0203440794115],
    all_points_y:[466.8333333333333,456.6075697211155,444.52257636122175,425.9302788844621,411.9860557768924,391.5345285524568,372.0126162018592,343.19455511288174,326.4614873837981,283.699203187251,263.2476759628154,246.5146082337317,233.5,227.9223107569721,219.5557768924303,213.04847277556442,201.89309428950864,187.9488711819389,178.6527224435591,164.7084993359894,154.48273572377158,149.83466135458167,136.82005312084993,130.31274900398407,123.8054448871182,97.77622841965471,69.88778220451528,50.36586985391767,38.280876494023914,40.140106241699876,49.43625498007969,57.802788844621524,57.802788844621524,66.16932270916335,81.04316069057106,104.2835325365206,121.01660026560425,143.3273572377158,163.7788844621514,176.79349269588315,187.9488711819389,201.89309428950864,215.83731739707835,215.83731739707835,226.06308100929616,242.7961487383798,257.66998671978746,269.7549800796812,280.910358565737,291.1361221779548,304.15073041168654,307.8691899070385,317.1653386454183,324.60225763612215,332.039176626826,350.6314741035856,370.15338645418325,382.238379814077,384.09760956175296,390.6049136786188,400.8306772908366,408.2675962815405,418.4933598937583,425.0006640106241,432.437583001328,439.8745019920318,447.3114209827357,460.3260292164674,466.99999999999994,466.8333333333333]
  };

  // Вычисление IoU
  const iou = calculateIoU(segmentationMap, polygon, width, height);
  console.log(`IoU: ${iou.toFixed(4)}`);

  // Отображение IoU в интерфейсе
  const statusMessage = document.getElementById("status-message");
  statusMessage.innerText += ` | IoU: ${iou.toFixed(4)}`;
};

const status = (message) => {
  const statusMessage = document.getElementById("status-message");
  statusMessage.innerText = message;
  console.log(message);
};

function logMemoryUsage() {
  if (performance.memory) {
    console.log(
        `Used JS Heap: ${(performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
    );
    console.log(
        `Total JS Heap: ${(performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
    );
    console.log(
        `Heap Limit: ${(performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`,
    );
  } else {
    console.warn("Performance memory API is not supported in this browser.");
  }
}

async function handleClick(event) {
  if (imageSegmenter === undefined) return;

  // Лог до сегментации
  logMemoryUsage();

  const start = performance.now();
  await imageSegmenter.segment(event.target, callback);
  const end = performance.now();
  console.log(`Latency (CPU): ${end - start} ms`);

  // Лог после сегментации
  logMemoryUsage();
}

const runPrediction = (modelName, input, initialisationStart) => {
  deeplab[modelName].then((model) => {
    model.segment(input).then((output) => {
      displaySegmentationMap(modelName, output);
      status(
          `Ran in ${(performance.now() - initialisationStart).toFixed(2)} ms`,
      );
      console.log("Memory usage after segmentation:");
    
      logMemoryUsage();
      

    });
  });
};

const runDeeplab = async (modelName) => {
  status(`Running the inference...`);
  const selector = document.getElementById("quantizationBytes");
  const quantizationBytes = Number(
      selector.options[selector.selectedIndex].text,
  );
  if (state.quantizationBytes !== quantizationBytes) {
    for (const base of modelNames) {
      if (deeplab[base]) {
        (await deeplab[base]).dispose();
        deeplab[base] = undefined;
      }
    }
    state.quantizationBytes = quantizationBytes;
  }
  const input = document.getElementById("input-image");
  if (!input.src || !input.src.length || input.src.length === 0) {
    status("Failed! Please load an image first.");
    return;
  }
  toggleInvisible("input-card", false);

  if (!deeplab[modelName]) {
    status("Loading the model...");
    const loadingStart = performance.now();
    deeplab[modelName] = load({ base: modelName, quantizationBytes });
    await deeplab[modelName];
    status(
        `Loaded the model in ${(
          (performance.now() - loadingStart) / 1000
        ).toFixed(2)} s`,
    );
  }
  const predictionStart = performance.now();
  // Лог до сегментации
  console.log("Memory usage before segmentation:");
  logMemoryUsage();
  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(modelName, input, predictionStart);
  } else {
    input.onload = () => {
      runPrediction(modelName, input, predictionStart);
    };
  }
  console.log("===============================");
};

window.onload = initializeModels;
