/**
 * jQuery json-viewer
 * @author: Alexandre Bodelot <alexandre.bodelot@gmail.com>
 * @link: https://github.com/abodelot/jquery.json-viewer
 */
(function($) {

  function isCollapsable(arg) {
    return arg instanceof Object && Object.keys(arg).length > 0;
  }

  function isUrl(string) {
    var protocols = ['http', 'https', 'ftp', 'ftps'];
    for (var i = 0; i < protocols.length; ++i) {
      if (string.startsWith(protocols[i] + '://')) {
        return true;
      }
    }
    return false;
  }

  function htmlEscape(s) {
    return s.replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/'/g, '&apos;')
      .replace(/"/g, '&quot;');
  }

  function json2html(json, options) {
    var html = '';
    if (typeof json === 'string') {
      json = htmlEscape(json);
      if (options.withLinks && isUrl(json)) {
        html += '<a href="' + json + '" class="json-string" target="_blank">' + json + '</a>';
      } else {
        json = json.replace(/&quot;/g, '\\&quot;');
        if (json.length > options.stringLengthThreshold) {
          var visiblePart = json.substring(0, options.stringLengthThreshold);
          visiblePart = visiblePart.replace(/\n/g, '<br>');
          var hiddenPart = json.substring(options.stringLengthThreshold);
          hiddenPart = hiddenPart.replace(/\n/g, '<br>');
          html += `<span class="json-string">"${visiblePart}<span class="json-more" style="display: none;">${hiddenPart}</span>"<a href="#" class="json-show-more">Show more...</a></span>`;
        } else {
          html += '<span class="json-string">"' + json + '"</span>';
        }
      }
    } else if (typeof json === 'number' || typeof json === 'bigint') {
      html += '<span class="json-literal">' + json + '</span>';
    } else if (typeof json === 'boolean') {
      html += '<span class="json-literal">' + json + '</span>';
    } else if (json === null) {
      html += '<span class="json-literal">null</span>';
    } else if (json instanceof Array) {
      if (json.length > 0) {
        html += '[<ol class="json-array">';
        for (var i = 0; i < json.length; ++i) {
          html += '<li>';
          if (isCollapsable(json[i])) {
            html += '<a href class="json-toggle"></a>';
          }
          html += json2html(json[i], options);
          if (i < json.length - 1) {
            html += ',';
          }
          html += '</li>';
        }
        html += '</ol>]';
      } else {
        html += '[]';
      }
    } else if (typeof json === 'object') {
      if (options.bigNumbers && (typeof json.toExponential === 'function' || json.isLosslessNumber)) {
        html += '<span class="json-literal">' + json.toString() + '</span>';
      } else {
        var keyCount = Object.keys(json).length;
        if (keyCount > 0) {
          html += '{<ul class="json-dict">';
          for (var key in json) {
            if (Object.prototype.hasOwnProperty.call(json, key)) {
              let jsonElement = json[key];
              key = htmlEscape(key);
              var keyRepr = options.withQuotes ? '<span class="json-string">"' + key + '"</span>' : key;
              html += '<li>';
              if (isCollapsable(jsonElement)) {
                html += '<a href class="json-toggle">' + keyRepr + '</a>';
              } else {
                html += keyRepr;
              }
              html += ': ' + json2html(jsonElement, options);
              if (--keyCount > 0) {
                html += ',';
              }
              html += '</li>';
            }
          }
          html += '</ul>}';
        } else {
          html += '{}';
        }
      }
    }
    return html;
  }

  $.fn.jsonViewer = function(json, options) {
    options = Object.assign({}, {
      collapsed: false,
      rootCollapsable: true,
      withQuotes: false,
      withLinks: true,
      bigNumbers: false,
      stringLengthThreshold: 50 // Default threshold for string length
    }, options);

    return this.each(function() {
      var html = json2html(json, options);
      if (options.rootCollapsable && isCollapsable(json)) {
        html = '<a href class="json-toggle"></a>' + html;
      }

      $(this).html(html);
      $(this).addClass('json-document');

      $(this).off('click');
      $(this).on('click', 'a.json-toggle', function() {
        var target = $(this).toggleClass('collapsed').siblings('ul.json-dict, ol.json-array');
        target.toggle();
        if (target.is(':visible')) {
          target.siblings('.json-placeholder').remove();
        } else {
          var count = target.children('li').length;
          var placeholder = count + (count > 1 ? ' items' : ' item');
          target.after('<a href class="json-placeholder">' + placeholder + '</a>');
        }
        return false;
      });

      // Handle click event for "Show more..." and "Hide" actions.
      $(this).on('click', '.json-show-more', function(e) {
        e.preventDefault();
        var $this = $(this);
        var $moreContent = $this.siblings('.json-more');
        $moreContent.toggle();
        if ($moreContent.is(':visible')) {
          $this.text('Hide');
        } else {
          $this.text('Show more...');
        }
      });

      if (options.collapsed == true) {
        $(this).find('a.json-toggle').first().click();
      }
    });
  };
})(jQuery);
