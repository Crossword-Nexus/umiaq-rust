/**
* Process a word list
**/

function processWordList(fileContents, minScore) {
  let lines = fileContents.toLowerCase().trim().replace(/\r/g, "").split("\n");

  // Only keep lines with a semicolon and a score >= minScore
  lines = lines.filter(line => {
    const parts = line.split(";");
    return parts.length === 2 && parseFloat(parts[1]) >= minScore;
  });

  // sort by length and then alphabetically
  lines.sort((wordA, wordB) => {
    return wordA.length - wordB.length || wordA.localeCompare(wordB);
  });

  // remove semicolon and score
  lines = lines.map(x => x.split(';')[0]);

  // add to window
  window.wordlist = lines;
}

async function getLocalWordList() {
  const url = '../data/spreadthewordlist.dict';
  let minScore = 50;

  try {
    const resp = await fetch(url, { cache: 'no-cache' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    processWordList(text, minScore);
  } catch (err) {
    console.error(err);
    alert(`Could not load "${url}". Check the path and CORS/MIME type.`);
  }
}


/**
* Modal boxes (currently only for wordlists)
**/

// Close the modal box
function closeModalBox() {
  const modal = document.getElementById('cw-modal');
  modal.style.display = 'none';
}

// Create a generic modal box with content
function createModalBox(title, content, button_text = 'Close') {

  // Set the contents of the modal box
  const modalContent = `
  <div class="modal-content">
    <div class="modal-header">
      <span class="modal-close" id="modal-close">&times;</span>
      <span class="modal-title">${title}</span>
    </div>
    <div class="modal-body">
      ${content}
    </div>
    <!-- No footer
    <div class="modal-footer">
      <button class="cw-button" id="modal-button">${button_text}</button>
    </div>
    -->
  </div>`;
  // Set this to be the contents of the container modal div
  const modal = document.getElementById('cw-modal');
  modal.innerHTML = modalContent;

  // Show the div
  modal.style.display = 'block';

  // Allow user to close the div
  const modalClose = document.getElementById('modal-close');

  // When the user clicks on <span> (x), close the modal
  modalClose.onclick = function () {
    closeModalBox();
  };
  // When the user clicks anywhere outside the modal, close it
  window.onclick = function (event) {
    if (event.target === modal) {
      closeModalBox();
    }
  };

  // Clicking the button should close the modal
    /* ... but we've removed that button
  const modalButton = document.getElementById('modal-button');
  modalButton.onclick = function () {
    closeModalBox();
  };
     */
}

/** Assign a click action to the word list button **/
function handleWordlistClick() {
  let files = document.getElementById('wordlist-file').files; // FileList object
  let minScore = document.getElementById('min-score').value;
  minScore = parseInt(minScore);

  // files is a FileList of File objects.
  for (let i = 0; i < files.length; i++) {
      let f = files[i];
      if (f) {
          let r = new FileReader();

          r.onload = (function () {
              return function (e) {
                  let contents = e.target.result;
                  processWordList(contents, minScore);
                  closeModalBox();
              };
          })(f);
          r.readAsText(f);
      } else {
          alert("Failed to load file");
      }
  }
}

function createWordlistModal() {
  let title = 'Upload your own word list';
  let html = `
  <input type="file" id="wordlist-file"  accept=".txt,.dict" />
  <label for="min-score">Min score:</label>
  <input type="number" id="min-score" name="min-score" value="50">
  <br /><br />
  <button value="Submit" class="button-primary" id="submit-wordlist">Upload</button>
  `;
  createModalBox(title, html);

  // add a listener to the submit button
  document.getElementById('submit-wordlist').addEventListener('click', handleWordlistClick);

}
