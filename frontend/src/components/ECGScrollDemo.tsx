import { ContainerScroll } from "@/components/ui/container-scroll-animation";

export function ECGScrollDemo() {
  return (
    <div className="flex flex-col overflow-hidden">
      <ContainerScroll
        titleComponent={
          <>
            <h1 className="text-4xl font-semibold text-black dark:text-white">
              Advanced ECG Analysis <br />
              <span className="text-4xl md:text-[6rem] font-bold mt-1 leading-none">
                CardioAI
              </span>
            </h1>
          </>
        }
      >
        <img
          src="https://ecglibrary.com/ecgs/norm.png"
          alt="ECG Analysis Dashboard"
          className="mx-auto rounded-2xl object-cover h-full object-left-top"
          draggable={false}
        />
      </ContainerScroll>
    </div>
  );
}
