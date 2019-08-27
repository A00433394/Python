template1 = """
<object>
        <image>{0}</image>
"""
template2 = """ <bndbox>
            <X>{0}</X>
            <Y>{1}</Y>
            <L>{2}</L>
            <W>{3}</W>
            <result>{4}<result>
    </bndbox>
"""

res = []


def process_file():

    # read file
    with open("one.txt") as input_:
        for line in input_:
            if(line.startswith("IMG")):
                res.append([])
            res[-1].append(line.strip())

    print(res[2])

    # convert to XML format

    with open("output_file.xml", "w+") as output_:
        output_.write("""<?xml version="1.0" encoding="utf-8"?>
<annotation>""")
        for items in res:
            output_.write(template1.format(items[0]))
            for x in items[1:]:
                x1, x2, x3, x4, x5 = x.split(",")
                output_.write(template2.format(x1, x2, x3, x4, x5))
            output_.write("</object>")
        output_.write("</annonations>")
    #  then add XML header manually


if __name__ == '__main__':
    process_file()

#    <object>
#         <image>IMG_2012-09-28T09-00-03.jpg</image>
#     <bndbox>
#             <x1>261</x1>
#             <y1>138</y1>
#             <x2>284</x2>
#             <y2>140</y2>
#             <result>u<result>
#     </bndbox>
# 	<bndbox>
#             <x1>261</x1>
#             <y1>138</y1>
#             <x2>284</x2>
#             <y2>140</y2>
#             <result>u<result>
#     </bndbox>
# 	<bndbox>
#             <x1>261</x1>
#             <y1>138</y1>
#             <x2>284</x2>
#             <y2>140</y2>
#             <result>u<result>
#     </bndbox>
#     </object>
