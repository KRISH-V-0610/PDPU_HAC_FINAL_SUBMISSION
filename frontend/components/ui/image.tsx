import NextImage from 'next/image';

interface ImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}

export function Image({ src, alt, width, height, className }: ImageProps) {
  return (
    <NextImage
      src={src}
      alt={alt}
      width={width}
      height={height}
      className={className}
    />
  );
} 