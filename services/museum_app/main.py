import flet as ft

import requests


url_pred = "http://46.191.235.91:8099/predict/"
url_desc = "http://46.191.235.91:8099/predict_desc_by_ai/"

key_map = {"id": "id: ", "name": "Имя объекта: \n", "description": "Описание объекта: \n", "object_id": "id объекта: ", "group": "Категория: ", "score": "Сходство: "}


def get_result(url, file_path):
    print(file_path)
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file, "image/jpeg")}
        response = requests.post(url, files=files)
    return response.json()


def main(page: ft.Page):

    def checkbox_changed(e):
        page.controls.insert(4, ft.Image(src="spinner.gif"))
        page.update()
        res = get_result(url_desc, selected_file_im)
        page.controls.pop(4)
        page.update()

        page.controls.insert(4, ft.Text("Описание предмета на изображении:", weight=ft.FontWeight.BOLD, size=25, color=ft.colors.BLACK))
        page.controls.insert(5, ft.Text(res['generated_by_ai'][0], size=20, color=ft.colors.BLACK))
        page.controls.pop(3)
        page.update()
        checkbox.on_change = None

    def handle_delete(e: ft.ControlEvent):
        panel.controls.remove(e.control.data)
        page.update()


    checkbox = ft.Checkbox(value=False, on_change=checkbox_changed)

    page.bgcolor = ft.colors.BROWN_100
    page.update()
    #img = ft.Image()

    def display_im(path):
        img = ft.Image(
            src=path,
            width=400,
            height=400,
            fit=ft.ImageFit.CONTAIN,
        )
        page.controls.append(img)
        page.update()

    def display_results(res):
        print(type(res))
        im_class = res["classifier"]["group"]
        page.controls.append(ft.Text("Класс изображения: ", color=ft.colors.BLACK, weight=ft.FontWeight.BOLD, size=25))
        page.controls.append(ft.Text(im_class, size=20, color=ft.colors.BLACK))
        page.controls.append(
            ft.Text("Наиболее похожие изображения: ", color=ft.colors.BLACK, weight=ft.FontWeight.BOLD, size=25))
        for res_i in res['topk']:
            exp = ft.ExpansionPanel(header=ft.Container(content=ft.Image(src_base64=res_i["bytes_image"]), margin=5, padding=5, alignment=ft.alignment.center, border_radius=5, bgcolor=ft.colors.BROWN_100,), can_tap_header=True, bgcolor=ft.colors.BROWN_100)
            spans = []
            for key, value in res_i.items():
                if key == "description" and value == "":
                    key = "img_name"
                if key != "bytes_image" and key != "img_name":
                    key = key_map[key]
                    spans.append(ft.TextSpan(key, ft.TextStyle(size=20, weight=ft.FontWeight.BOLD)))
                    spans.append(ft.TextSpan(str(value)+'\n', ft.TextStyle(size=17)))
            exp.content = ft.ListTile(
                title=ft.Text(spans=spans),
                trailing=ft.IconButton(ft.icons.DELETE, on_click=handle_delete, data=exp),
                bgcolor=ft.colors.BROWN_100,
                text_color=ft.colors.BLACK,
            )
            panel.controls.append(exp)

        page.controls.append(panel)
        page.update()

    def pick_files_result(e: ft.FilePickerResultEvent):
        global panel
        panel = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.BLACK,
            elevation=8,
            divider_color=ft.colors.BLACK,
        )

        if len(page.controls) > 2:
            for i in range (len(page.controls) - 2):
                page.controls.pop()
            page.update()
        selected_file_text.value = (
             ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
         )
        selected_file_text.update()
        global selected_file_im
        selected_file_im = e.files[0].path
        display_im(selected_file_im)
        checkbox = ft.ElevatedButton(text="Сгенерировать описание",
                                     on_click=checkbox_changed, color=ft.colors.BROWN_600, bgcolor=ft.colors.WHITE)
        page.controls.append(checkbox)
        results = get_result(url_pred, selected_file_im)
        page.scroll = True
        display_results(results)
        return selected_file_im


    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_file_text = ft.Text(color=ft.colors.BLACK)

    page.overlay.append(pick_files_dialog)
    page.add(ft.Text("", size=5))
    page.add(
        ft.Row(
            [
                ft.ElevatedButton(
                    "Загрузить изображение",
                    icon=ft.icons.UPLOAD_FILE,
                    color=ft.colors.BROWN_600,
                    bgcolor=ft.colors.WHITE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False,
                        allowed_extensions=['png', 'jpg', 'jpeg'],
                    ),
                ),
                selected_file_text,
            ]
        )
    )


ft.app(target=main)
