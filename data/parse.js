module.exports = function parseTextBody(sentence) {
    var text = "";
    // Eliminate everything with URLs
    if (sentence.indexOf('http') === -1) {
        var text = sentence.trim().toLowerCase();
        // Add punctuation if there isn 't any
        if (text.charAt(text.length - 1) !== '\.' && text.charAt(text.length - 1) !== '\!' && text.charAt(text.length - 1) !== '\?') {
            text = text.concat('\.');
        }
        //Remove everything inside parentheses and brackets
        text = text.replace(/\[([^\[\]]*)\]/g, '').replace(/\(([^\(\)]+)\)/g, '');

        //Fix "I" capitalization
        text = text.replace(/\si(?=[\s\.\!\?])/g, " I");
        text = text.replace(/\si\'m(?=[\s\.\!\?])/g, " I'm");
        text = text.replace(/\si\'ve(?=[\s\.\!\?])/g, " I've");
        text = text.replace(/\si\'d(?=[\s\.\!\?])/g, " I'd");
        text = text.replace(/\si\'ll(?=[\s\.\!\?])/g, " I'll");
        text = text.replace(/\sim(?=[\s\.\!\?])/g, " I'm");
        text = text.replace(/\sive(?=[\s\.\!\?])/g, " I've");
        text = text.replace(/\sid(?=[\s\.\!\?])/g, " I'd");
        text = text.replace(/\sill(?=[\s\.\!\?])/g, " I'll");

        text = text.replace(/^i(?=[\s\.\!\?])/g, "I");
        text = text.replace(/^i\'m(?=[\s\.\!\?])/g, "I'm");
        text = text.replace(/^i\'ve(?=[\s\.\!\?])/g, "I've");
        text = text.replace(/^i\'d(?=[\s\.\!\?])/g, "I'd");
        text = text.replace(/^i\'ll(?=[\s\.\!\?])/g, "I'll");
        text = text.replace(/^im(?=[\s\.\!\?])/g, "I'm");
        text = text.replace(/^ive(?=[\s\.\!\?])/g, "I've");
        text = text.replace(/^id(?=[\s\.\!\?])/g, "I'd");
        text = text.replace(/^ill(?=[\s\.\!\?])/g, "I'll");

        //Remove newlines (nor sure why there would be any)
        text = text.replace(/\n/g, " ");

        text = text.replace(/(\.)\1{1,}/g, ".").replace(/(\!)\1{1,}/g, "!").replace(/(\?)\1{1,}/g, "?");
        text = text.replace(/\.\s\.\s\./g, ".").replace(/\!\s\!\s\!/g, "!").replace(/\?\s\?\s\?/g, "?")
        text = text.replace(/mrs\./g, 'mrs ').replace(/mr\./g, 'mr ').replace(/ms\./g, 'ms ').replace(/dr\./g, 'dr ').replace(/♪/g, '').replace(/�/g, '').replace(/(\s)\1{1,}/g, " ").replace(/(\_)\1{1,}/g, '')

        //FUCKING HASHTAGS
        text = text.replace(/#\w+/g, 'fuck');

        //GRAMMARZ, not shed or wed...
        text = text.replace(/\su(?=[\s\.\!\?])/g, " you");
        text = text.replace(/\suve(?=[\s\.\!\?])/g, " you've");
        text = text.replace(/\sud(?=[\s\.\!\?])/g, " you'd");
        text = text.replace(/\sull(?=[\s\.\!\?])/g, " you'll");
        text = text.replace(/\suve(?=[\s\.\!\?])/g, " you've");
        text = text.replace(/\shed(?=[\s\.\!\?])/g, " he'd");
        text = text.replace(/\sitd(?=[\s\.\!\?])/g, " it'd");
        text = text.replace(/\shed(?=[\s\.\!\?])/g, " he'd");
        text = text.replace(/\stheyd(?=[\s\.\!\?])/g, " they'd");
        text = text.replace(/\sthatd(?=[\s\.\!\?])/g, " that'd");
        text = text.replace(/\swhod(?=[\s\.\!\?])/g, " who'd");
        text = text.replace(/\swhatd(?=[\s\.\!\?])/g, " what'd");
        text = text.replace(/\swhered(?=[\s\.\!\?])/g, " where'd");
        text = text.replace(/\swhend(?=[\s\.\!\?])/g, " when'd");
        text = text.replace(/\swhyd(?=[\s\.\!\?])/g, " why'd");
        text = text.replace(/\showd(?=[\s\.\!\?])/g, " how'd");
        text = text.replace(/\shes(?=[\s\.\!\?])/g, " he's");
        text = text.replace(/\sshes(?=[\s\.\!\?])/g, " she's");
        text = text.replace(/\stheres(?=[\s\.\!\?])/g, " there's");
        text = text.replace(/\sisnt(?=[\s\.\!\?])/g, " isn't");
        text = text.replace(/\sarent(?=[\s\.\!\?])/g, " aren't");
        text = text.replace(/\swasnt(?=[\s\.\!\?])/g, " wasn't");
        text = text.replace(/\swerent(?=[\s\.\!\?])/g, " weren't");
        text = text.replace(/\shavent(?=[\s\.\!\?])/g, " haven't");
        text = text.replace(/\shasnt(?=[\s\.\!\?])/g, " hasn't");
        text = text.replace(/\shadnt(?=[\s\.\!\?])/g, " hadn't");
        text = text.replace(/\swont(?=[\s\.\!\?])/g, " won't");
        text = text.replace(/\swouldnt(?=[\s\.\!\?])/g, " wouldn't");
        text = text.replace(/\sdont(?=[\s\.\!\?])/g, " don't");
        text = text.replace(/\sdoesnt(?=[\s\.\!\?])/g, " doesn't");
        text = text.replace(/\sdidnt(?=[\s\.\!\?])/g, " didn't");
        text = text.replace(/\scant(?=[\s\.\!\?])/g, " can't");
        text = text.replace(/couldnt(?=[\s\.\!\?])/g, "couldn't");
        text = text.replace(/shouldnt(?=[\s\.\!\?])/g, "shouldn't");
        text = text.replace(/mightnt(?=[\s\.\!\?])/g, "mightn't");
        text = text.replace(/mustnt(?=[\s\.\!\?])/g, "mustn't");
        text = text.replace(/wouldve(?=[\s\.\!\?])/g, "would've");
        text = text.replace(/shouldve(?=[\s\.\!\?])/g, "should've");
        text = text.replace(/couldve(?=[\s\.\!\?])/g, "could've");
        text = text.replace(/mightve(?=[\s\.\!\?])/g, "might've");
        text = text.replace(/mustve(?=[\s\.\!\?])/g, "must've");

        text = text.replace(/^u(?=[\s\.\!\?])/g, "you");
        text = text.replace(/^uve(?=[\s\.\!\?])/g, "you've");
        text = text.replace(/^ud(?=[\s\.\!\?])/g, "you'd");
        text = text.replace(/^ull(?=[\s\.\!\?])/g, "you'll");
        text = text.replace(/^uve(?=[\s\.\!\?])/g, "you've");
        text = text.replace(/^hed(?=[\s\.\!\?])/g, "he'd");
        text = text.replace(/^itd(?=[\s\.\!\?])/g, "it'd");
        text = text.replace(/^hed(?=[\s\.\!\?])/g, "he'd");
        text = text.replace(/^theyd(?=[\s\.\!\?])/g, "they'd");
        text = text.replace(/^thatd(?=[\s\.\!\?])/g, "that'd");
        text = text.replace(/^whod(?=[\s\.\!\?])/g, "who'd");
        text = text.replace(/^whatd(?=[\s\.\!\?])/g, "what'd");
        text = text.replace(/^whered(?=[\s\.\!\?])/g, "where'd");
        text = text.replace(/^whend(?=[\s\.\!\?])/g, "when'd");
        text = text.replace(/^whyd(?=[\s\.\!\?])/g, "why'd");
        text = text.replace(/^howd(?=[\s\.\!\?])/g, "how'd");
        text = text.replace(/^hes(?=[\s\.\!\?])/g, "he's");
        text = text.replace(/^shes(?=[\s\.\!\?])/g, "she's");
        text = text.replace(/^theres(?=[\s\.\!\?])/g, "there's");
        text = text.replace(/^isnt(?=[\s\.\!\?])/g, "isn't");
        text = text.replace(/^arent(?=[\s\.\!\?])/g, "aren't");
        text = text.replace(/^wasnt(?=[\s\.\!\?])/g, "wasn't");
        text = text.replace(/^werent(?=[\s\.\!\?])/g, "weren't");
        text = text.replace(/^havent(?=[\s\.\!\?])/g, "haven't");
        text = text.replace(/^hasnt(?=[\s\.\!\?])/g, "hasn't");
        text = text.replace(/^hadnt(?=[\s\.\!\?])/g, "hadn't");
        text = text.replace(/^wont(?=[\s\.\!\?])/g, "won't");
        text = text.replace(/^wouldnt(?=[\s\.\!\?])/g, "wouldn't");
        text = text.replace(/^dont(?=[\s\.\!\?])/g, "don't");
        text = text.replace(/^doesnt(?=[\s\.\!\?])/g, "doesn't");
        text = text.replace(/^didnt(?=[\s\.\!\?])/g, "didn't");
        text = text.replace(/^cant(?=[\s\.\!\?])/g, "can't");



        //Remove the rest of the garbage
        text = text.replace(/[^\w\s\.\!\,\?\']/g, '');
        text = text.trim()

        return text
    }
}
