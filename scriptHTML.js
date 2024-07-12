import {loadAll, predict} from "/script.js"

loadAll();

const submitButton = document.getElementById("submitButton");
const inputElements = document.querySelectorAll(".form-control");
submitButton.addEventListener("click", predictPrice);

function predictPrice() {
    let inputFeatures = new Array(20);

    for (let i = 0; i < inputElements.length; i++) {
        let val = inputElements[i].value;
        if (val == '') {
            alert("Please enter all the features");
            return;
        }
        inputFeatures[i] = parseFloat(val);
    }

    let result = predict(inputFeatures);
    if (!result) {
        alert("The model is still loading!");
    }
    else {
        result.array().then(res => {
            let roundedRes = Math.round(res);
            if (roundedRes == 0) alert("Very Low Price");
            if (roundedRes == 1) alert("Low Price");
            if (roundedRes == 2) alert("Medium Price");
            if (roundedRes >= 3) alert("High Price");
        })
    }
}