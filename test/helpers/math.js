module.exports = {

  roundTo: function(num, decimals) {
    if (typeof decimals === 'undefined') decimals = 2;
    return Number(num.toFixed(2));
  }

};
