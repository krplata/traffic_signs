from yattag import Doc


class HtmlGenerator:
    def __init__(self, image_data=None):
        self.__image_data__ = image_data
        self.__doc__, self.__tag__, self.__text__ = Doc().tagtext()
        self.__full_doc__()

    def __body__(self):
        with self.__tag__('body'):
            for path, title in self.__image_data__:
                with self.__tag__('div', klass='image'):
                    with self.__tag__('h2', klass='title'):
                        self.__text__(title)
                    self.__doc__.asis(f"<img src=\"{path}\", alt=\"image\"/>")

    def __head__(self):
        with self.__tag__('head'):
            self.__doc__.asis('<meta charset=\'UTF-8\'/>')
            self.__doc__.asis(
                '<link rel=\"stylesheet\" type=\"text/css\" href=\"styles.css\"/>')

    def __full_doc__(self):
        with self.__tag__('html'):
            self.__head__()
            self.__body__()

    def get_html_template(self):
        return self.__doc__.getvalue()

    def dump_html_to_file(self, filepath):
        with open(filepath, 'w') as file:
            file.write(self.__doc__.getvalue())
